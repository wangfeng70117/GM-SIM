/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}

torch::Tensor
KNNCUDA(const torch::Tensor& points, const int N)
{
	const int P = points.size(0);
	auto float_opts = points.options().dtype(torch::kFloat32);
	auto int_opts = points.options().dtype(torch::kInt32);
	torch::Tensor nearestIds = torch::full({ P, N }, 0, int_opts);
	SimpleKNN::getNearestIds(P, (float3*)points.contiguous().data<float>(), nearestIds.contiguous().data<int>(), N);
	return nearestIds;
}