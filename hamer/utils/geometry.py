from typing import Optional
import torch
from torch.nn import functional as F

def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Uses the direct normalized-quaternion formula; avoids 9 separate scalar
    products and a large torch.stack call.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    # Compute all products in batch; far fewer Python-level ops than unpacking 9 scalars
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rotMat = torch.stack([
        1 - 2*(yy + zz),  2*(xy - wz),      2*(xz + wy),
        2*(xy + wz),      1 - 2*(xx + zz),  2*(yz - wx),
        2*(xz - wy),      2*(yz + wx),      1 - 2*(xx + yy)
    ], dim=1).view(B, 3, 3)
    return rotMat


def rotmat_to_aa(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to axis-angle vectors.
    Args:
        rotmat (torch.Tensor): Tensor of shape (B, 3, 3).
    Returns:
        torch.Tensor: Axis-angle vectors of shape (B, 3).
    """
    batch_size = rotmat.shape[0]
    trace = rotmat[:, 0, 0] + rotmat[:, 1, 1] + rotmat[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)

    axis = torch.stack([
        rotmat[:, 2, 1] - rotmat[:, 1, 2],
        rotmat[:, 0, 2] - rotmat[:, 2, 0],
        rotmat[:, 1, 0] - rotmat[:, 0, 1],
    ], dim=1)

    sin_theta = torch.sin(theta)
    small_angle = sin_theta.abs() < 1e-4
    scale = torch.empty_like(theta)
    scale[~small_angle] = theta[~small_angle] / (2.0 * sin_theta[~small_angle])
    scale[small_angle] = 0.5
    aa = axis * scale.unsqueeze(1)

    if small_angle.any():
        identity = torch.eye(3, device=rotmat.device, dtype=rotmat.dtype).unsqueeze(0)
        delta = rotmat[small_angle] - identity
        aa[small_angle] = torch.stack([
            delta[:, 2, 1] - delta[:, 1, 2],
            delta[:, 0, 2] - delta[:, 2, 0],
            delta[:, 1, 0] - delta[:, 0, 1],
        ], dim=1) * 0.5

    return aa.view(batch_size, 3)


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.

    Optimized vs. original:
    - Skips the identity rotation einsum when rotation=None (saves one batched matmul per call).
    - Replaces K-matrix allocation + two einsums with direct component-wise multiply.
    - Uses slicing [:, :, -1:] instead of unsqueeze(-1) for the depth division.

    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]

    # Only apply rotation when it is not the identity — saves one batched matmul per call
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation, points)

    # Translate into camera space
    points = points + translation.unsqueeze(1)   # (B, N, 3)

    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)

    # Perspective divide — use slicing to avoid a new tensor from unsqueeze
    z = points[:, :, 2:3]                        # (B, N, 1)
    xy = points[:, :, :2] / z                    # (B, N, 2)  — normalized image-plane coords

    # Apply intrinsics component-wise; avoids allocating a (B, 3, 3) K matrix
    # and doing two einsums.  focal_length: (B, 2), camera_center: (B, 2)
    projected = xy * focal_length.unsqueeze(1) + camera_center.unsqueeze(1)  # (B, N, 2)

    return projected
