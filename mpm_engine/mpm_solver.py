import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from mpm_engine.export_utils import *
from mpm_engine.warp_utils import *
from mpm_engine.mpm_utils import *


class MPM_Simulator_WARP:
    def __init__(self, n_particles, n_grid=128, grid_lim=1.0, device="cuda:0"):
        self.initialize(n_particles, n_grid, grid_lim, device=device)
        self.time_profile = {}

    def initialize(self, n_particles, n_grid=128, grid_lim=1.0, device="cuda:0"):
        self.n_particles = n_particles
        self.mpm_model = MPMModelStruct()
        self.mpm_model.grid_lim = grid_lim
        self.mpm_model.n_grid = n_grid
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        self.mpm_model.dx = self.mpm_model.grid_lim / self.mpm_model.n_grid
        self.mpm_model.inv_dx = float(self.mpm_model.n_grid / self.mpm_model.grid_lim)
        # MPM的参数是存储在每个粒子上的
        self.mpm_model.E = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.nu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.mu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.lam = wp.zeros(shape=n_particles, dtype=float, device=device)

        # 是否根据形变梯度更新协方差矩阵
        self.mpm_model.update_cov_with_F = False

        # material is used to switch between different elastoplastic models. 0 is jelly
        self.mpm_model.material = 0

        self.mpm_model.plastic_viscosity = 0.0
        self.mpm_model.softening = 0.1
        self.mpm_model.yield_stress = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_model.friction_angle = 25.0
        sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
        self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)
        self.mpm_model.rpic_damping = 0.0  # 0.0 if no damping (apic). -1 if pic
        self.mpm_model.grid_v_damping_scale = 1.1  # globally applied

        self.mpm_state = MPMStateStruct()
        self.mpm_state.particle_x = wp.empty(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # current position

        self.mpm_state.particle_v = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity

        self.mpm_state.particle_F = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle F elastic

        self.mpm_state.particle_R = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle R rotation

        self.mpm_state.particle_init_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # initial covariance matrix

        self.mpm_state.particle_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # current covariance matrix

        self.mpm_state.particle_F_trial = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # apply return mapping will yield

        self.mpm_state.particle_stress = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )

        self.mpm_state.particle_vol = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle volume
        self.mpm_state.particle_mass = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle mass
        self.mpm_state.particle_density = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_state.particle_C = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_Jp = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )

        self.mpm_state.particle_selection = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )

        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_mark = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=int,
            device=device,
        )
        self.time = 0.0

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []


    def import_particle_x_from_torch(self, tensor_x, clone=True, device="cuda:0"):
        if tensor_x is not None:
            if clone:
                tensor_x = tensor_x.clone().detach()
            self.mpm_state.particle_x = torch2warp_vec3(tensor_x, dvc=device)

    def load_initial_data_from_torch(
            self,
            tensor_x,
            tensor_volume,
            particle_selected,
            tensor_cov=None,
            n_grid=128,
            grid_lim=1,
            device="cuda:0"
    ):
        self.dim, self.n_particles = tensor_x.shape[1], tensor_x.shape[0]
        assert tensor_x.shape[0] == tensor_volume.shape[0]
        self.initialize(self.n_particles, n_grid, grid_lim, device)
        # 将tensor的粒子位置加载到MPM的Warp张量中
        self.import_particle_x_from_torch(tensor_x)
        self.mpm_state.particle_vol = wp.from_numpy(
            tensor_volume.detach().clone().cpu().numpy(), dtype=float, device=device
        )
        self.mpm_state.particle_selection = wp.from_numpy(
            particle_selected.detach().clone().cpu().numpy().astype(int),
            dtype=int,
            device=device
        )

        # 仿真物体的协方差矩阵， 传入协方差矩阵并且随着形变梯度更新协方差矩阵时运行。
        if tensor_cov is not None:
            self.mpm_state.particle_init_cov = wp.from_numpy(
                tensor_cov.reshape(-1).detach().clone().cpu().numpy(),
                dtype=float,
                device=device,
            )

            if self.mpm_model.update_cov_with_F:
                self.mpm_state.particle_cov = self.mpm_state.particle_init_cov
        # 将粒子的初始速度设置为0
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # 将粒子的协方差矩阵设置为单位矩阵
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        with wp.ScopedTimer(
                "mark_grid", synchronize=True, print=False, dict=self.time_profile
        ):
            # 设置不进行仿真的网格部分
            wp.launch(
                kernel=mark_grid,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model],
            )

        print("Particles initialized from torch data.")
        print("Total particles: ", self.n_particles)

    def set_parameters_dict(self, kwargs={}, device="cuda:0"):
        if "material" in kwargs:
            if kwargs["material"] == "jelly":
                self.mpm_model.material = 0
            elif kwargs["material"] == "metal":
                self.mpm_model.material = 1
            elif kwargs["material"] == "sand":
                self.mpm_model.material = 2
            elif kwargs["material"] == "foam":
                self.mpm_model.material = 3
            elif kwargs["material"] == "snow":
                self.mpm_model.material = 4
            elif kwargs["material"] == "plasticine":
                self.mpm_model.material = 5
            else:
                raise TypeError("Undefined material type")

        if "grid_lim" in kwargs:
            self.mpm_model.grid_lim = kwargs["grid_lim"]

        if "n_grid" in kwargs:
            self.mpm_model.n_grid = kwargs["n_grid"]

        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )
        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )

        if "E" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.E, kwargs["E"]],
                device=device,
            )
        if "nu" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.nu, kwargs["nu"]],
                device=device,
            )
        if "yield_stress" in kwargs:
            val = kwargs["yield_stress"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.yield_stress, val],
                device=device,
            )
        if "hardening" in kwargs:
            self.mpm_model.hardening = kwargs["hardening"]
        if "xi" in kwargs:
            self.mpm_model.xi = kwargs["xi"]
        if "friction_angle" in kwargs:
            self.mpm_model.friction_angle = kwargs["friction_angle"]
            sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
            self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        if "g" in kwargs:
            self.mpm_model.gravitational_accelaration = wp.vec3(
                kwargs["g"][0], kwargs["g"][1], kwargs["g"][2]
            )

        if "density" in kwargs:
            density_value = kwargs["density"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_density, density_value],
                device=device,
            )
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )
        if "rpic_damping" in kwargs:
            self.mpm_model.rpic_damping = kwargs["rpic_damping"]
        if "plastic_viscosity" in kwargs:
            self.mpm_model.plastic_viscosity = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            self.mpm_model.softening = kwargs["softening"]
        if "grid_v_damping_scale" in kwargs:
            self.mpm_model.grid_v_damping_scale = kwargs["grid_v_damping_scale"]

        if "additional_material_params" in kwargs:
            for params in kwargs["additional_material_params"]:
                param_modifier = MaterialParamsModifier()
                param_modifier.point = wp.vec3(params["point"])
                param_modifier.size = wp.vec3(params["size"])
                param_modifier.density = params["density"]
                param_modifier.E = params["E"]
                param_modifier.nu = params["nu"]
                wp.launch(
                    kernel=apply_additional_params,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, param_modifier],
                    device=device,
                )

            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )

    def finalize_mu_lam(self, device="cuda:0"):
        wp.launch(
            kernel=compute_mu_lam_from_E_nu,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )

    def add_object_velocity(self, v, object_mask, device="cuda:0"):
        spacial_particles_mask = wp.from_numpy(
            object_mask.cpu().numpy(),
            dtype=int,
            device=device
        )
        wp.launch(
            kernel=add_init_velocity,
            dim=self.n_particles,
            inputs=[self.mpm_state, v, spacial_particles_mask],
            device=device
        )


    def set_special_object(self, param, object_mask, device="cuda:0"):
        print(f'param is {param}, {type(object_mask)}')
        spacial_particles_mask = wp.from_numpy(
            object_mask.cpu().numpy(),
            dtype=int,
            device=device
        )

        if "E" in param and "nu" in param:
            wp.launch(
                kernel=apply_spacial_params,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, param["E"], param["nu"], spacial_particles_mask],
                device=device,
            )
        else:
            raise NotImplementedError("Please input the special E and nu")

    def init_particle_mark(self, particle_mark, device="cuda:0"):
        self.mpm_state.particle_mark = wp.zeros(shape=particle_mark.shape[0], dtype=int, device=device)

    def add_object_mark(self, object_mask, mark_number, device="cuda:0"):
        mark_wp = wp.from_numpy(
            object_mask.detach().clone().cpu().numpy().astype(int),
            dtype=int,
            device=device
        )
        print(f'n_particles is {self.n_particles}, object_mask.shape is {object_mask.shape}, mark_wp.shape is {mark_wp.shape}')
        print(f'mark_number is {mark_number}')
        wp.launch(
            kernel=set_mark_number,
            dim=self.n_particles,
            inputs=[self.mpm_state, mark_wp, mark_number],
            device=device,
        )


    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky",
        friction=0.0,
        start_time=0.0,
        end_time=999.0,
    ):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])

        if surface == "sticky" and friction != 0:
            raise ValueError("friction must be 0 on sticky surfaces.")
        if surface == "sticky":
            collider_param.surface_type = 0
        elif surface == "slip":
            collider_param.surface_type = 1
        elif surface == "cut":
            collider_param.surface_type = 11
        else:
            collider_param.surface_type = 2
        # frictional
        collider_param.friction = friction

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )
                    elif param.surface_type == 11:
                        if (
                            float(grid_z) * model.dx < 0.4
                            or float(grid_z) * model.dx > 0.53
                        ):
                            state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                                0.0, 0.0, 0.0
                            )
                        else:
                            v_in = state.grid_v_out[grid_x, grid_y, grid_z]
                            state.grid_v_out[grid_x, grid_y, grid_z] = (
                                wp.vec3(v_in[0], 0.0, v_in[2]) * 0.3
                            )
                    else:
                        v = state.grid_v_out[grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        if param.surface_type == 1:
                            v = (
                                v - normal_component * n
                            )  # Project out all normal component
                        else:
                            v = (
                                v - wp.min(normal_component, 0.0) * n
                            )  # Project out only inward normal component
                        if normal_component < 0.0 and wp.length(v) > 1e-20:
                            v = wp.max(
                                0.0, wp.length(v) + normal_component * param.friction
                            ) * wp.normalize(
                                v
                            )  # apply friction here
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    def p2g2p(self, step, dt, device="cuda:0"):
        if step == 50:
            self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z
        )
        # 清空网格速度
        wp.launch(
            kernel=zero_grid,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model],
            device=device
        )

        # compute stress = stress(returnMap(F_trial))
        # 根据形变梯度计算粒子内部的应力
        # 给state.particle_F和state.particle_stress赋值
        with wp.ScopedTimer(
            "compute_stress_from_F_trial",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # F and stress are updated
        # 给粒子一个速度
        # wp.launch(
        #     kernel=modify_particle_v_before_p2g,
        #     dim=self.n_particles,
        #     inputs=[self.mpm_state, wp.vec3(-0.2, -0.08, 0), 0.0, 0.0, 0.0]
        # )

        # p2g
        with wp.ScopedTimer(
            "p2g",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # apply p2g'

        # grid update
        with wp.ScopedTimer(
                "grid_update", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )

        if self.mpm_model.grid_v_damping_scale < 1.0:
            wp.launch(
                kernel=add_damping_via_grid,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model.grid_v_damping_scale],
                device=device,
            )

        # apply BC on grid
        with wp.ScopedTimer(
                "apply_BC_on_grid", synchronize=True, print=False, dict=self.time_profile
        ):
            for k in range(len(self.grid_postprocess)):
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[
                        self.time,
                        dt,
                        self.mpm_state,
                        self.mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])

            # g2p
        with wp.ScopedTimer(
                "g2p", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=g2p,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt, step],
                device=device,
            )  # x, v, C, F_trial are updated

        self.time = self.time + dt




    def export_particle_x_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_x)

    def export_particle_cov_to_torch(self, device="cuda:0"):
        if not self.mpm_model.update_cov_with_F:
            with wp.ScopedTimer(
                "compute_cov_from_F",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=compute_cov_from_F,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model],
                    device=device,
                )

        cov = wp.to_torch(self.mpm_state.particle_cov)
        return cov

    def export_particle_R_to_torch(self, device="cuda:0"):
        with wp.ScopedTimer(
            "compute_R_from_F",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_R_from_F,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

        R_tensor = wp.to_torch(self.mpm_state.particle_R)
        R_tensor = R_tensor.reshape(-1, 9)
        return R_tensor

    def print_time_profile(self):
        print("MPM Time profile:")
        for key, value in self.time_profile.items():
            print(key, sum(value))