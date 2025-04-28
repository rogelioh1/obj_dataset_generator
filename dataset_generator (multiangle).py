import os
import re
import numpy as np
from PIL import Image, ImageDraw
import open3d as o3d
from scipy.ndimage import gaussian_filter, binary_fill_holes
from math import ceil

def extract_id_from_filename(filename):
    """Extract identity ID from the filename."""
    match = re.search(r'real_(\d+)', filename)
    if match:
        return match.group(1)
    return None
    
def process_obj_file(obj_path, output_dir, identity_id, camera_angles=None):
    """Process OBJ file to extract RGB, depth, normal map, and point cloud data from multiple angles."""
    # Define default camera angles if not provided
    if camera_angles is None:
        # Format: (azimuth, elevation) in degrees
        camera_angles = [
            (0, 0),      # Front
            (30, 0),     # Right 30°
            (-30, 0),    # Left 30°
            (0, 15),     # Up 15°
            (0, -15),    # Down 15°
            (45, 0),     # Right 45°
            (-45, 0),    # Left 45°
            (30, 15),    # Right 30° up 15°
            (-30, 15),   # Left 30° up 15°
            (30, -15),   # Right 30° down 15°
            (-30, -15),  # Left 30° down 15°
        ]
    
    # Create output directories if they don't exist
    rgb_dir = os.path.join(output_dir, identity_id, 'rgb')
    depth_dir = os.path.join(output_dir, identity_id, 'depth')
    normal_dir = os.path.join(output_dir, identity_id, 'normal')
    pointcloud_dir = os.path.join(output_dir, identity_id, 'pointcloud')
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(pointcloud_dir, exist_ok=True)
    
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(obj_path))[0]
    
    # Process each camera angle
    for angle_idx, (azimuth, elevation) in enumerate(camera_angles):
        angle_suffix = f"_a{azimuth}_e{elevation}"
        
        # 1. Generate synthetic RGB image
        rgb_path = os.path.join(rgb_dir, f"{base_filename}{angle_suffix}.png")
        generate_rgb(obj_path, rgb_path, azimuth=azimuth, elevation=elevation)

        # 2. Generate camera-like depth map
        depth_path = os.path.join(depth_dir, f"{base_filename}{angle_suffix}.png")
        generate_clean_depth_map(obj_path, depth_path, azimuth=azimuth, elevation=elevation)
        
        # 3. Generate normal map from OBJ
        normal_path = os.path.join(normal_dir, f"{base_filename}{angle_suffix}.png")
        generate_normal_map(obj_path, normal_path, azimuth=azimuth, elevation=elevation)
        
        # 4. Generate point cloud data (only for front view to save space)
        if angle_idx == 0:  # Only for front view
            pointcloud_path = os.path.join(pointcloud_dir, f"{base_filename}.ply")
            generate_point_cloud(obj_path, pointcloud_path)
            
        print(f"Processed {base_filename} - view {angle_idx+1}/{len(camera_angles)}: azimuth={azimuth}, elevation={elevation}")
    
    print(f"Processed {base_filename} for identity {identity_id} with {len(camera_angles)} camera angles")


def get_rotation_matrix(azimuth, elevation):
    """
    Get rotation matrix for a given azimuth and elevation in degrees.
    Azimuth rotates around Y axis (left-right).
    Elevation rotates around X axis (up-down).
    """
    # Convert to radians
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    
    # Create rotation matrices
    rot_y = np.array([
        [np.cos(azimuth_rad), 0, np.sin(azimuth_rad)],
        [0, 1, 0],
        [-np.sin(azimuth_rad), 0, np.cos(azimuth_rad)]
    ])
    
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(elevation_rad), -np.sin(elevation_rad)],
        [0, np.sin(elevation_rad), np.cos(elevation_rad)]
    ])
    
    # Combine rotations (order matters: first azimuth, then elevation)
    return np.dot(rot_x, rot_y)

def generate_rgb(obj_path, output_path, width=256, height=256, azimuth=0, elevation=0):
    """Generate an RGB image using a direct mesh manipulation approach."""
    try:
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(obj_path)
        
        # Get vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Apply rotation to fix orientation - USE THE ORIGINAL ROTATION
        rotation = np.array([
            [1, 0, 0],  # Original X
            [0, -1, 0],  # Flip Y
            [0, 0, -1]   # Flip Z
        ])
        vertices = np.dot(vertices, rotation)

        # Apply camera rotation based on azimuth and elevation
        camera_rotation = get_rotation_matrix(azimuth, elevation)
        vertices = np.dot(vertices, camera_rotation)
        
        # Center the mesh
        mesh_center = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
        vertices = vertices - mesh_center
        
        # Scale to fit the image
        max_dim = np.max(np.abs(vertices))
        scale_factor = 1.0 * min(width, height) / (2 * max_dim)
        vertices = vertices * scale_factor
        
        # Create an image with white background
        rgb_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Try to get textures or vertex colors
        has_textures = hasattr(mesh, 'textures') and len(mesh.textures) > 0
        has_colors = hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0
        
        # Default face color if no texture
        default_color = np.array([210, 170, 140])  # Medium skin tone
        
        # Create depth buffer for z-sorting
        depth_buffer = np.ones((height, width)) * float('inf')
        mask = np.zeros((height, width), dtype=bool)
        
        # Simple lighting setup
        light_dir = np.array([0, 0, 1])  # Light from front
        ambient = 0.3
        diffuse = 0.7
        
        # Project vertices to screen space and render
        for triangle in triangles:
            v0, v1, v2 = vertices[triangle]
            
            # Project to screen space (scale and center)
            x0, y0, z0 = v0
            x1, y1, z1 = v1
            x2, y2, z2 = v2
            
            # Flip y and convert to pixel coordinates
            x0 = int(width / 2 + x0)
            y0 = int(height / 2 - y0)
            x1 = int(width / 2 + x1)
            y1 = int(height / 2 - y1)
            x2 = int(width / 2 + x2)
            y2 = int(height / 2 - y2)
            
            # Ensure coordinates are within bounds
            x0 = max(0, min(width-1, x0))
            y0 = max(0, min(height-1, y0))
            x1 = max(0, min(width-1, x1))
            y1 = max(0, min(height-1, y1))
            x2 = max(0, min(width-1, x2))
            y2 = max(0, min(height-1, y2))
            
            # Calculate face normal for lighting
            v0_v1 = np.array([x1-x0, y1-y0, z1-z0])
            v0_v2 = np.array([x2-x0, y2-y0, z2-z0])
            normal = np.cross(v0_v1, v0_v2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            
            # Skip if face is facing away (simple backface culling)
            if normal[2] <= 0:
                continue
            
            # Calculate lighting
            diffuse_factor = max(0, np.dot(normal, light_dir))
            lighting = ambient + diffuse * diffuse_factor
            
            # Get triangle color
            if has_colors:
                triangle_color = np.mean([mesh.vertex_colors[i] for i in triangle], axis=0) * 255
            else:
                triangle_color = default_color
            
            # Apply lighting
            lit_color = triangle_color * lighting
            lit_color = np.clip(lit_color, 0, 255).astype(np.uint8)
            
            # Use PIL for triangle rasterization
            img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img)
            draw.polygon([(x0, y0), (x1, y1), (x2, y2)], fill=1)
            triangle_mask = np.array(img) > 0
            
            # Fill triangle
            y_coords, x_coords = np.where(triangle_mask)
            for i in range(len(y_coords)):
                y, x = y_coords[i], x_coords[i]
                
                # Calculate barycentric coordinates for proper interpolation
                area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
                if abs(area) < 1e-10:
                    continue
                
                s = 0.5 * ((y0 - y2) * (x - x2) + (x2 - x0) * (y - y2)) / area
                t = 0.5 * ((y2 - y1) * (x - x2) + (x1 - x2) * (y - y2)) / area
                u = 1 - s - t
                
                # Interpolate z value for depth test
                z = u * z2 + s * z0 + t * z1
                
                # Update if this point is closer
                if z < depth_buffer[y, x]:
                    depth_buffer[y, x] = z
                    rgb_img[y, x] = lit_color
                    mask[y, x] = True
        
        # Fill holes
        filled_mask = binary_fill_holes(mask)
        
        # Apply smoothing to the face (not the background)
        temp_img = rgb_img.copy().astype(float)
        for c in range(3):
            channel = temp_img[:,:,c]
            smoothed = gaussian_filter(channel, sigma=0.7)
            rgb_img[:,:,c] = np.where(filled_mask, smoothed, rgb_img[:,:,c])
        
        # Convert to PIL image for rotation
        rgb_pil = Image.fromarray(rgb_img.astype(np.uint8))
        
        # Apply 180 degree rotation to fix orientation
        rgb_pil = rgb_pil.rotate(180)

        #Apply horizontal flip to match depth map orientation
        rgb_pil = rgb_pil.transpose(Image.FLIP_LEFT_RIGHT)

        # Save the image
        rgb_pil.save(output_path)
        print(f"Generated RGB image with corrected orientation: {output_path}")
        
        # Also rotate the mask
        rotated_mask = np.rot90(filled_mask, k=2)
        
        return rotated_mask
    
    except Exception as e:
        print(f"Error in RGB generation: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a simple fallback image
        rgb_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        Image.fromarray(rgb_img).save(output_path)
        print(f"Created blank image due to error: {output_path}")
        
        return np.zeros((height, width), dtype=bool)
                        
def generate_clean_depth_map(obj_path, output_path, width=256, height=256, azimuth=0, elevation=0):
    """Generate a clean depth map with no noise artifacts,
    where closer parts are darker and the background is white."""
    try:
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(obj_path)
        
        # Get vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Apply rotation to fix orientation - USE THE SAME ROTATION AS IN RGB
        rotation = np.array([
            [1, 0, 0],  # Original X
            [0, -1, 0],  # Flip Y
            [0, 0, -1]   # Flip Z
        ])
        vertices = np.dot(vertices, rotation)

                # Apply camera rotation based on azimuth and elevation
        camera_rotation = get_rotation_matrix(azimuth, elevation)
        vertices = np.dot(vertices, camera_rotation)
        
        # Center the mesh - MATCH RGB GENERATION
        mesh_center = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
        vertices = vertices - mesh_center
        
        # Scale to fit the image - MATCH RGB GENERATION
        max_dim = np.max(np.abs(vertices))
        scale_factor = 1.0 * min(width, height) / (2 * max_dim)
        vertices = vertices * scale_factor
        
        # Create depth buffer
        depth_buffer = np.ones((height, width)) * float('inf')
        mask = np.zeros((height, width), dtype=bool)
        
        # Project vertices to screen space
        for triangle in triangles:
            v0, v1, v2 = vertices[triangle]
            
            # Project to screen space (scale and center) - MATCH RGB GENERATION
            x0, y0, z0 = v0
            x1, y1, z1 = v1
            x2, y2, z2 = v2
            
            # Flip y and convert to pixel coordinates
            x0 = int(width / 2 + x0)
            y0 = int(height / 2 - y0)
            x1 = int(width / 2 + x1)
            y1 = int(height / 2 - y1)
            x2 = int(width / 2 + x2)
            y2 = int(height / 2 - y2)

            # Use PIL for triangle rasterization
            img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img)
            draw.polygon([(x0, y0), (x1, y1), (x2, y2)], fill=1)
            triangle_mask = np.array(img) > 0
            
            # Process each pixel in the triangle
            y_coords, x_coords = np.where(triangle_mask)
            for i in range(len(y_coords)):
                y, x = y_coords[i], x_coords[i]
                
                # Calculate barycentric coordinates
                area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
                if abs(area) < 1e-10:
                    continue
                
                s = 0.5 * ((y0 - y2) * (x - x2) + (x2 - x0) * (y - y2)) / area
                t = 0.5 * ((y2 - y1) * (x - x2) + (x1 - x2) * (y - y2)) / area
                u = 1 - s - t
                
                # Interpolate z value for depth test
                z = u * z2 + s * z0 + t * z1
                
                # Update if this point is closer
                if z < depth_buffer[y, x]:
                    depth_buffer[y, x] = z
                    mask[y, x] = True
        
        # Clean up the mask with hole filling
        mask = binary_fill_holes(mask)
        
        # Set the background to white (255)
        depth_img = np.ones((height, width), dtype=np.uint8) * 255
        
        if np.any(mask):
            # Replace inf values with finite values before calculating min/max
            valid_depth = depth_buffer[mask]
            valid_depth = valid_depth[np.isfinite(valid_depth)]
            
            if len(valid_depth) > 0:
                min_val = np.min(valid_depth)
                max_val = np.max(valid_depth)
                depth_range = max_val - min_val
                
                if depth_range > 1e-10:
                    # Create a clean dark face with depth variations
                    valid_indices = mask & np.isfinite(depth_buffer)
                    normalized_depth = np.zeros_like(depth_buffer)
                    normalized_depth[valid_indices] = (depth_buffer[valid_indices] - min_val) / depth_range
                    
                    # Closer points (smaller depth values) should be darker
                    # Scale to 20-100 range for better contrast
                    depth_values = np.zeros_like(depth_img, dtype=float)
                    depth_values[valid_indices] = 20 + normalized_depth[valid_indices] * 80
                    depth_img[valid_indices] = np.clip(depth_values[valid_indices], 0, 255).astype(np.uint8)
                else:
                    # Depth range is too small, use a flat value
                    depth_img[mask] = 60
            else:
                # No valid depth values, use a flat value
                depth_img[mask] = 60
            
            # Apply mild smoothing to create more natural depth transitions
            # But avoid smoothing over the edges between foreground and background
            foreground = depth_img < 255
            
            # Create a smoothed version
            smoothed = gaussian_filter(depth_img.astype(float), sigma=0.7)
            
            # Only apply smoothing to the foreground
            depth_img = np.where(foreground, smoothed, depth_img).astype(np.uint8)
            
            # Apply 180 degree rotation to fix orientation and flip horizontally
            # to match the RGB processing
            depth_pil = Image.fromarray(depth_img.astype(np.uint8))
            depth_pil = depth_pil.rotate(180)
            depth_pil = depth_pil.transpose(Image.FLIP_LEFT_RIGHT)
            depth_pil.save(output_path)
            
            print(f"Generated clean depth map with no artifacts: {output_path}")
            
            # Return cleaned depth buffer
            clean_depth = depth_buffer.copy()
            clean_depth[~np.isfinite(clean_depth)] = max_val if 'max_val' in locals() else 0
            
            return clean_depth, mask
        else:
            # No valid mask, create an empty depth map
            print(f"Generated empty depth map: {output_path}")
            return depth_buffer, mask
        
    except Exception as e:
        print(f"Error generating depth map for {obj_path}: {e}")
        # Create a default depth map in case of error
        depth_img = np.ones((height, width), dtype=np.uint8) * 255
        img = Image.fromarray(depth_img)
        img.save(output_path)
        print(f"Created default depth map due to error: {output_path}")
        return None, None
                
def generate_point_cloud(obj_path, output_path, depth_buffer=None, mask=None, width=256, height=256):
    """Generate a point cloud PLY file from the 3D model."""
    try:
        # If depth_buffer and mask weren't provided, generate them
        if depth_buffer is None or mask is None:
            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(obj_path)
            
            # Get vertices and triangles
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            # Apply a rotation to fix orientation
            rotation = np.array([
                [1, 0, 0],
                [0, -1, 0],  # Flip Y
                [0, 0, -1]   # Flip Z
            ])
            vertices = np.dot(vertices, rotation)
            
            # Create empty point cloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(vertices)
            
            # Try to get vertex colors if available
            if hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0:
                point_cloud.colors = mesh.vertex_colors
            else:
                # Try to load the corresponding PNG for color
                png_path = os.path.join(os.path.dirname(obj_path), f"{os.path.splitext(os.path.basename(obj_path))[0]}.png")
                if os.path.exists(png_path):
                    # For simplicity, just use a uniform color
                    point_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
                else:
                    point_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
            
        else:
            # Get mesh for dimension reference
            mesh = o3d.io.read_triangle_mesh(obj_path)
            vertices = np.asarray(mesh.vertices)
            
            if len(vertices) == 0:
                print(f"No vertices found in mesh: {obj_path}")
                return
                
            min_x, min_y, min_z = np.min(vertices, axis=0)
            max_x, max_y, max_z = np.max(vertices, axis=0)
            
            # Add random noise to make it look like a real depth sensor
            noise_scale = (max_z - min_z) * 0.01  # 1% of depth range
            
            # Create point cloud from depth buffer
            points = []
            valid_mask = mask & np.isfinite(depth_buffer)
            
            # Convert depth buffer to points
            for y in range(height):
                for x in range(width):
                    if valid_mask[y, x]:
                        # Convert image coordinates to world coordinates
                        nx = (x / (width - 1)) * (max_x - min_x) + min_x
                        ny = (y / (height - 1)) * (max_y - min_y) + min_y
                        nz = depth_buffer[y, x]
                        
                        # Add some noise to z to simulate sensor noise
                        nz += np.random.normal(0, noise_scale)
                        
                        points.append([nx, ny, nz])
            
            # Skip if no points
            if len(points) == 0:
                print(f"No valid points found for point cloud: {obj_path}")
                return
            
            # Create point cloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
            point_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
        
        # Make sure we have points
        if len(np.asarray(point_cloud.points)) == 0:
            print(f"No points in point cloud: {obj_path}")
            return
        
        # Randomly downsample to simulate real depth sensor sparsity
        # Keep about 20% of points for sparse point cloud
        point_cloud = point_cloud.random_down_sample(0.2)
        
        # Add noise to points to make it look realistic
        points = np.asarray(point_cloud.points)
        if len(points) > 0:  # Make sure we still have points after downsampling
            # Get depth range for noise scaling
            min_depth = np.min(points[:, 2])
            max_depth = np.max(points[:, 2])
            depth_range = max_depth - min_depth
            
            if depth_range > 0:  # Avoid division by zero or very small numbers
                noise = np.random.normal(0, depth_range * 0.005, size=points.shape)  # 0.5% of depth range
                points += noise
                point_cloud.points = o3d.utility.Vector3dVector(points)
            
            # Only remove outliers if we have enough points
            if len(points) > 30:  # Need at least enough points for statistics
                # Remove statistical outliers
                point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=min(20, len(points)-1), std_ratio=2.0)
            
            # Save the point cloud
            o3d.io.write_point_cloud(output_path, point_cloud)
            print(f"Generated point cloud: {output_path}")
        else:
            print(f"Point cloud empty after downsampling: {obj_path}")
    
    except Exception as e:
        print(f"Error generating point cloud for {obj_path}: {e}")

def generate_normal_map(obj_path, output_path, width=256, height=256, azimuth=0, elevation=0):
    """Generate a normal map with smooth gradients that aligns with RGB images."""
    try:
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(obj_path)
        
        # Get vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Apply rotation to fix orientation - MATCH RGB GENERATION
        rotation = np.array([
            [1, 0, 0],  # Original X
            [0, -1, 0],  # Flip Y
            [0, 0, -1]   # Flip Z
        ])
        vertices = np.dot(vertices, rotation)

        # Apply camera rotation based on azimuth and elevation
        camera_rotation = get_rotation_matrix(azimuth, elevation)
        vertices = np.dot(vertices, camera_rotation)
        
        # Center the mesh - MATCH RGB GENERATION
        mesh_center = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
        vertices = vertices - mesh_center
        
        # Scale to fit the image - MATCH RGB GENERATION
        max_dim = np.max(np.abs(vertices))
        scale_factor = 1.0 * min(width, height) / (2 * max_dim)
        vertices = vertices * scale_factor
        
        # Apply mesh smoothing
        temp_mesh = o3d.geometry.TriangleMesh()
        temp_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        temp_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        temp_mesh = temp_mesh.filter_smooth_taubin(number_of_iterations=5)
        temp_mesh.compute_vertex_normals()
        
        vertices = np.asarray(temp_mesh.vertices)
        triangles = np.asarray(temp_mesh.triangles)
        normals = np.asarray(temp_mesh.vertex_normals)
        
        # Create normal buffer and depth buffer
        normal_buffer = np.zeros((height, width, 3), dtype=np.float32)
        depth_buffer = np.ones((height, width)) * float('inf')
        
        # Project vertices to screen space
        for triangle in triangles:
            v0, v1, v2 = vertices[triangle]
            n0, n1, n2 = normals[triangle]
            
            # Project to screen space (scale and center) - MATCH RGB GENERATION
            x0, y0, z0 = v0
            x1, y1, z1 = v1
            x2, y2, z2 = v2
            
            # Flip y and convert to pixel coordinates - MATCH RGB GENERATION
            x0 = int(width / 2 + x0)
            y0 = int(height / 2 - y0)
            x1 = int(width / 2 + x1)
            y1 = int(height / 2 - y1)
            x2 = int(width / 2 + x2)
            y2 = int(height / 2 - y2)
            
            # Ensure coordinates are within bounds
            x0 = max(0, min(width-1, x0))
            y0 = max(0, min(height-1, y0))
            x1 = max(0, min(width-1, x1))
            y1 = max(0, min(height-1, y1))
            x2 = max(0, min(width-1, x2))
            y2 = max(0, min(height-1, y2))
            
            # Use PIL for clean triangle rasterization
            img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img)
            draw.polygon([(x0, y0), (x1, y1), (x2, y2)], fill=1)
            triangle_mask = np.array(img) > 0
            
            # Process each pixel in the triangle
            y_indices, x_indices = np.where(triangle_mask)
            for i in range(len(y_indices)):
                y, x = y_indices[i], x_indices[i]
                
                # Calculate barycentric coordinates
                area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
                if abs(area) < 1e-10:
                    continue
                
                s = 0.5 * ((y0 - y2) * (x - x2) + (x2 - x0) * (y - y2)) / area
                t = 0.5 * ((y2 - y1) * (x - x2) + (x1 - x2) * (y - y2)) / area
                u = 1 - s - t
                
                # Interpolate z value for depth test
                z = u * z2 + s * z0 + t * z1
                
                # Update if this point is closer
                if z < depth_buffer[y, x]:
                    depth_buffer[y, x] = z
                    
                    # Interpolate normal
                    normal = u * n2 + s * n0 + t * n1
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm  # Normalize
                    
                    # Store normal (convert from [-1,1] to [0,1] range)
                    normal_buffer[y, x] = (normal + 1) / 2
        
        # Get the face mask
        mask = depth_buffer != float('inf')
        mask = binary_fill_holes(mask)
        
        # Apply slight smoothing to the normal map
        for i in range(3):
            normal_buffer[:,:,i] = gaussian_filter(normal_buffer[:,:,i], sigma=0.5)
        
        # Apply mask to normal buffer
        mask_3d = np.stack([mask] * 3, axis=2)
        normal_buffer = normal_buffer * mask_3d
        
        # Enhance contrast slightly
        for i in range(3):
            channel = normal_buffer[:,:,i]
            if np.max(channel) > np.min(channel):
                channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
                normal_buffer[:,:,i] = channel
        
        # Ensure values are in [0,1] range
        normal_buffer = np.clip(normal_buffer, 0, 1)
        
        # Convert to RGB image
        normal_img = (normal_buffer * 255).astype(np.uint8)
        
        # Apply 180 degree rotation to fix orientation - MATCH RGB GENERATION
        normal_pil = Image.fromarray(normal_img)
        normal_pil = normal_pil.rotate(180)
        
        # Apply horizontal flip to match RGB orientation - MATCH RGB GENERATION
        normal_pil = normal_pil.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Save the normal map
        normal_pil.save(output_path)
        print(f"Generated normal map: {output_path}")
        
    except Exception as e:
        print(f"Error generating normal map for {obj_path}: {e}")
        # Create a default normal map
        img = Image.new('RGB', (width, height), color=(128, 128, 255))
        img.save(output_path)
        print(f"Created default normal map due to error: {output_path}")

def process_batch(obj_files, batch_index, batch_size, source_dir, output_dir):
    """Process a batch of OBJ files."""
    start_idx = batch_index * batch_size
    end_idx = min(start_idx + batch_size, len(obj_files))
    
    print(f"Processing batch {batch_index+1}: files {start_idx+1} to {end_idx} of {len(obj_files)}")
    
    for i in range(start_idx, end_idx):
        obj_file = obj_files[i]
        obj_path = os.path.join(source_dir, obj_file)
        
        # Extract identity ID from filename
        identity_id = extract_id_from_filename(obj_file)
        if identity_id is None:
            print(f"Skipping {obj_file}: Could not extract identity ID")
            continue
        
        # Process the OBJ file and organize the data
        process_obj_file(obj_path, output_dir, identity_id)
        print(f"Processed {i+1}/{len(obj_files)}: {obj_file} for identity {identity_id}")

def main():
    # Windows-style path for source directory containing the OBJ, MAT, and PNG files
    source_dir = r"C:\\depth\\Dataset_Generator\\obj_files_test"  # Update this path
    
    # Windows-style path for output directory for the organized dataset
    output_dir = r"C:\\depth\\Dataset_Generator\\output"  # Update this path
    
    # Define camera angles for multi-view dataset
    camera_angles = [
        (0, 0),      # Front
        (30, 0),     # Right 30°
        (-30, 0),    # Left 30°
        (0, 15),     # Up 15°
        (0, -15),    # Down 15°
        (45, 0),     # Right 45°
        (-45, 0),    # Left 45°
        (30, 15),    # Right 30° up 15°
        (-30, 15),   # Left 30° up 15°
        (30, -15),   # Right 30° down 15°
        (-30, -15),  # Left 30° down 15°
    ]
    
    # Batch processing parameters
    batch_size = 5  # Process files in batches to manage memory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all OBJ files in the source directory
    obj_files = [f for f in os.listdir(source_dir) if f.endswith('.obj')]
    total_files = len(obj_files)
    
    print(f"Found {total_files} OBJ files in {source_dir}")
    print(f"Organizing files into dataset structure in {output_dir} with {len(camera_angles)} camera angles")
    
    # Calculate number of batches
    num_batches = ceil(total_files / batch_size)
    
    # Process files in batches
    for batch_index in range(num_batches):
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, len(obj_files))
        
        print(f"Processing batch {batch_index+1}: files {start_idx+1} to {end_idx} of {len(obj_files)}")
        
        for i in range(start_idx, end_idx):
            obj_file = obj_files[i]
            obj_path = os.path.join(source_dir, obj_file)
            
            # Extract identity ID from filename
            identity_id = extract_id_from_filename(obj_file)
            if identity_id is None:
                print(f"Skipping {obj_file}: Could not extract identity ID")
                continue
            
            # Process the OBJ file from multiple camera angles
            process_obj_file(obj_path, output_dir, identity_id, camera_angles)
            print(f"Processed {i+1}/{len(obj_files)}: {obj_file} for identity {identity_id}") 
    
    print("Dataset organization completed!")
    print(f"Total files processed: {total_files}")
    print(f"Total images generated: {total_files * len(camera_angles)}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()