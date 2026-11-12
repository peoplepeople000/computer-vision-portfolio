"""R28 camera-aware NLF geometry. Existing assumed-FOV poses are controls only."""
import numpy as np
import field_context as crop
import nlf_pose_features as nlf


def crop_affine(box, width, height):
    """Original image -> frozen rounded224 letterbox, including pixel centers."""
    basis=crop.crop_to_image([[0.,0.],[1.,0.],[0.,1.]],box,width,height)
    if not np.isfinite(basis).all():raise ValueError('Invalid crop')
    inverse=np.array([[basis[1,0]-basis[0,0],0.,basis[0,0]],
                      [0.,basis[2,1]-basis[0,1],basis[0,1]],[0.,0.,1.]])
    assert inverse[0,0]>0 and inverse[1,1]>0
    return np.linalg.inv(inverse)


def camera_inputs(calibration, box, frame_index):
    """Identity extrinsic keeps NLF output in the original camera axes."""
    if calibration['frame_index']!=frame_index or not calibration['quality']['passed']:return None
    c=calibration['parameters']['cam_params'];r=np.asarray(c['rotation_matrix'],float)
    np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-3);assert abs(np.linalg.det(r)-1)<1e-3
    k=np.array([[c['x_focal_length'],0,c['principal_point'][0]],
                [0,c['y_focal_length'],c['principal_point'][1]],[0,0,1.]],float)
    a=crop_affine(box,calibration['width'],calibration['height']);kcrop=a@k
    up=r@np.array([0.,0.,-1.]);up/=np.linalg.norm(up)
    assert kcrop[0,0]>0 and kcrop[1,1]>0 and np.isfinite(kcrop).all()
    return {'intrinsic':kcrop,'full_intrinsic':k,'affine':a,'rotation':r,
            'camera_up':up,'goal_side':calibration['quality']['unique_goal_side']}


def project_camera(points, intrinsic):
    points=np.asarray(points,float);valid=np.isfinite(points).all(-1)&(points[...,2]>1e-6)
    homogeneous=points@np.asarray(intrinsic).T
    uv=np.full((*points.shape[:-1],2),np.nan)
    uv[valid]=homogeneous[valid,:2]/homogeneous[valid,2:]
    return uv,valid


def camera_to_field(points, rotation, side):
    """Relative axes: kicker-view left, toward goal, up. No ground intersection."""
    assert side in (-1,1)
    world=np.asarray(points,float)@np.asarray(rotation,float)
    return np.stack([-side*world[...,1],side*world[...,0],-world[...,2]],-1)


def reflect_field(points, uncertainty):
    return np.asarray(points)[nlf.JOINT_SWAP]*[-1.,1.,1.],np.asarray(uncertainty)[nlf.JOINT_SWAP]


def field_features(points, uncertainty, agreement, rotation, side, mirrored=False):
    # The original camera-basis implementation already applies the intended
    # root/torso normalization and uncertainty gate without a body-axis rotation.
    transformed=camera_to_field(points,rotation,side)
    if mirrored:transformed,uncertainty=reflect_field(transformed,uncertainty)
    return nlf.geometry(transformed,uncertainty,agreement,'camera')
