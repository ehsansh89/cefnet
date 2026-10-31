import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import CRS, Transformer
import numpy as np

def create_region_grid(input_point, input_crs, point_type, grid_size, 
                        resolution_km, output_crs, 
                        row_col_convention='geospatial',
                        rotation_deg=0.0):
    """
    Create a grid of points on earth surface using UTM projection.
    """
    
    def get_local_utm_crs(lon, lat):
        """Get appropriate UTM CRS for a given lon/lat point."""
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = 'north' if lat >= 0 else 'south'
        epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
        return f'EPSG:{epsg_code}'
    
    def rotate_point_around_origin(x, y, angle_deg):
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_rot = x * cos_a + y * sin_a
        y_rot = -x * sin_a + y * cos_a
        return x_rot, y_rot
    
    # Main logic
    x, y = input_point
    nx, ny = grid_size
    resolution_m = resolution_km * 1000
    
    # Calculate total dimensions
    width_km = (nx - 1) * resolution_km
    height_km = (ny - 1) * resolution_km
    width_m = width_km * 1000
    height_m = height_km * 1000
    
    # Get input CRS
    crs_in = CRS.from_string(input_crs)
    
    # Convert input point to lat/lon if not already
    if not crs_in.is_geographic:
        transformer_to_geo = Transformer.from_crs(input_crs, 'EPSG:4326', 
                                                  always_xy=True)
        lon, lat = transformer_to_geo.transform(x, y)
    else:
        lon, lat = x, y
    
    # Get local UTM CRS for accurate metric calculations
    local_utm = get_local_utm_crs(lon, lat)
    
    # Transform input point to local UTM
    transformer_to_utm = Transformer.from_crs(input_crs, local_utm, 
                                              always_xy=True)
    utm_x, utm_y = transformer_to_utm.transform(x, y)
    
    # Calculate offsets
    if point_type == 'lower_left':
        input_offset_x = 0
        input_offset_y = 0
    elif point_type == 'lower_right':
        input_offset_x = width_m
        input_offset_y = 0
    elif point_type == 'upper_left':
        input_offset_x = 0
        input_offset_y = height_m
    elif point_type == 'upper_right':
        input_offset_x = width_m
        input_offset_y = height_m
    else:  # center
        input_offset_x = width_m / 2
        input_offset_y = height_m / 2
    
    # Create grid in UTM
    if row_col_convention == 'geospatial':
        # Meteorological convention: row 0 at top (north)
        utm_grid_points = []
        for i in range(ny):  # rows
            for j in range(nx):  # cols
                point_x = j * resolution_m 
                point_y = height_m - i * resolution_m 
                utm_grid_points.append((point_x, point_y, j, i))
    else:  # cartesian
        # Cartesian convention: row 0 at bottom (south)
        utm_grid_points = []
        for i in range(ny):  # rows
            for j in range(nx):  # cols
                point_x = j * resolution_m 
                point_y = i * resolution_m 
                utm_grid_points.append((point_x, point_y, j, i))
    
    # Apply rotation and translate
    rotated_utm_points = []
    for grid_x, grid_y, col, row in utm_grid_points:
        rel_x = grid_x - input_offset_x
        rel_y = grid_y - input_offset_y
        rot_x, rot_y = rotate_point_around_origin(rel_x, rel_y, rotation_deg)
        abs_x = utm_x + rot_x
        abs_y = utm_y + rot_y
        rotated_utm_points.append((abs_x, abs_y, col, row))
    
    # Transform to output CRS
    transformer_to_output = Transformer.from_crs(local_utm, output_crs, 
                                                 always_xy=True)
    output_points = []
    for utm_x, utm_y, col, row in rotated_utm_points:
        out_x, out_y = transformer_to_output.transform(utm_x, utm_y)
        output_points.append((out_x, out_y, col, row))
    
    # Create GeoDataFrame
    geometries = [Point(x, y) for x, y, _, _ in output_points]
    cols = [col for _, _, col, _ in output_points]
    rows = [row for _, _, _, row in output_points]
    
    gdf = gpd.GeoDataFrame({
        'geometry': geometries,
        'col': cols,
        'row': rows,
        'convention': [row_col_convention] * len(geometries),
        'res_km': [resolution_km] * len(geometries),
        'rot_deg': [rotation_deg] * len(geometries),
        'utm_crs': [local_utm] * len(geometries)  # <--- FIX: Store the CRS used!
    }, crs=output_crs)
    
    return gdf

def dilate_grid(gdf_grid, row_north=0, row_south=0, col_west=0, col_east=0):
    """
    Dilate (expand) an existing grid by adding rows and columns.
    """
    
    if len(gdf_grid) == 0:
        raise ValueError("Input grid is empty")
    
    # Extract grid parameters
    resolution_km = gdf_grid['res_km'].iloc[0]
    rotation_deg = gdf_grid['rot_deg'].iloc[0]
    convention = gdf_grid['convention'].iloc[0]
    output_crs = gdf_grid.crs
    
    # FIX: Retrieve the original CRS if available
    if 'utm_crs' in gdf_grid.columns:
        local_utm = gdf_grid['utm_crs'].iloc[0]
    else:
        # Fallback for old grids: Use the CENTROID instead of corner
        # This is safer than (0,0) but storing 'utm_crs' is the real fix.
        mean_lon = gdf_grid.geometry.x.mean()
        mean_lat = gdf_grid.geometry.y.mean()
        utm_zone = int((mean_lon + 180) / 6) + 1
        hemisphere = 'north' if mean_lat >= 0 else 'south'
        epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
        local_utm = f'EPSG:{epsg_code}'

    # Get original grid dimensions
    original_cols = gdf_grid['col'].max() + 1
    original_rows = gdf_grid['row'].max() + 1
    
    # Calculate new grid dimensions
    new_cols = original_cols + col_west + col_east
    new_rows = original_rows + row_north + row_south
    
    # Transform all original points to the CONSISTENT local_utm
    transformer_to_utm = Transformer.from_crs(output_crs, local_utm, always_xy=True)
    transformer_to_output = Transformer.from_crs(local_utm, output_crs, always_xy=True)
    
    # Build a mapping of (row, col) -> UTM coordinates
    utm_coords = {}
    for idx, row_data in gdf_grid.iterrows():
        row, col = row_data['row'], row_data['col']
        lon, lat = row_data.geometry.x, row_data.geometry.y
        utm_x, utm_y = transformer_to_utm.transform(lon, lat)
        utm_coords[(row, col)] = (utm_x, utm_y)
    
    # Calculate grid vectors
    def rotate_point_around_origin(x, y, angle_deg):
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_rot = x * cos_a + y * sin_a
        y_rot = -x * sin_a + y * cos_a
        return x_rot, y_rot
    
    resolution_m = resolution_km * 1000
    
    # Calculate unit vectors
    col_vec_x, col_vec_y = rotate_point_around_origin(resolution_m, 0, rotation_deg)
    
    if convention == 'geospatial':
        row_vec_x, row_vec_y = rotate_point_around_origin(0, -resolution_m, rotation_deg)
    else:  # cartesian
        row_vec_x, row_vec_y = rotate_point_around_origin(0, resolution_m, rotation_deg)
    
    # Generate new grid points
    new_points = []
    
    # We still need a reference point for extrapolation, 
    # but now we are guaranteed to be in the correct CRS.
    if (0, 0) in utm_coords:
        ref_utm_x, ref_utm_y = utm_coords[(0, 0)]
    else:
        ref_utm_x, ref_utm_y = list(utm_coords.values())[0]

    for new_row in range(new_rows):
        for new_col in range(new_cols):
            original_row = new_row - row_north
            original_col = new_col - col_west
            
            if (0 <= original_row < original_rows and 
                0 <= original_col < original_cols):
                # Use existing point (re-transforming to ensure consistency)
                utm_x, utm_y = utm_coords[(original_row, original_col)]
            else:
                # Extrapolate
                delta_row = new_row - row_north 
                delta_col = new_col - col_west  
                
                utm_x = ref_utm_x + delta_col * col_vec_x + delta_row * row_vec_x
                utm_y = ref_utm_y + delta_col * col_vec_y + delta_row * row_vec_y
            
            out_x, out_y = transformer_to_output.transform(utm_x, utm_y)
            new_points.append({
                'geometry': Point(out_x, out_y),
                'col': new_col,
                'row': new_row,
                'convention': convention,
                'res_km': resolution_km,
                'rot_deg': rotation_deg,
                'utm_crs': local_utm  # Propagate the correct CRS
            })
    
    gdf_dilated = gpd.GeoDataFrame(new_points, crs=output_crs)
    return gdf_dilated

def refine_grid_2x(gdf_grid):
    """
    Refines an input grid by doubling the shape (2x rows, 2x cols) 
    and halving the resolution.
    
    Logic:
    - New Resolution = Old Resolution / 2
    - New Width = 2 * Old Width
    - New Height = 2 * Old Height
    - Points are interpolated between existing points.
    - An extra layer of points is added at the end (half distance outward).
    """
    if len(gdf_grid) == 0:
        raise ValueError("Input grid is empty")

    # 1. Extract Metadata
    # We take the first row's metadata as representative
    meta = gdf_grid.iloc[0]
    old_res_km = meta['res_km']
    rotation_deg = meta['rot_deg']
    convention = meta['convention']
    local_utm = meta['utm_crs']
    output_crs = gdf_grid.crs

    # 2. Calculate New Parameters
    new_res_km = old_res_km / 2.0
    new_res_m = new_res_km * 1000.0
    
    # Determine grid dimensions based on indices to handle potential missing points in input
    min_row, max_row = gdf_grid['row'].min(), gdf_grid['row'].max()
    min_col, max_col = gdf_grid['col'].min(), gdf_grid['col'].max()
    
    n_rows_old = max_row - min_row + 1
    n_cols_old = max_col - min_col + 1
    
    n_rows_new = n_rows_old * 2
    n_cols_new = n_cols_old * 2

    # 3. Setup Coordinate Transformations
    # We must calculate in the original local UTM to ensure geometric consistency
    transformer_to_utm = Transformer.from_crs(output_crs, local_utm, always_xy=True)
    transformer_to_output = Transformer.from_crs(local_utm, output_crs, always_xy=True)

    # 4. Find the Anchor Point (Top-Left / Origin of the current grid indices)
    # We identify the point at (min_row, min_col). 
    # If strictly rectangular, it exists. If sparse, we project based on any valid point.
    # To be robust, we find the point corresponding to min_row/min_col.
    mask_origin = (gdf_grid['row'] == min_row) & (gdf_grid['col'] == min_col)
    
    if mask_origin.any():
        origin_geom = gdf_grid.loc[mask_origin, 'geometry'].iloc[0]
        origin_utm_x, origin_utm_y = transformer_to_utm.transform(origin_geom.x, origin_geom.y)
    else:
        # Fallback: calculate anchor from the first available point
        first = gdf_grid.iloc[0]
        fx, fy = transformer_to_utm.transform(first.geometry.x, first.geometry.y)
        
        # We need the OLD vectors to backtrack to the virtual origin
        old_res_m = old_res_km * 1000
        
        def simple_rotate(x, y, a_deg):
            a_rad = np.radians(a_deg)
            cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)
            return x * cos_a + y * sin_a, -x * sin_a + y * cos_a

        old_vec_x_col, old_vec_y_col = simple_rotate(old_res_m, 0, rotation_deg)
        if convention == 'geospatial':
            old_vec_x_row, old_vec_y_row = simple_rotate(0, -old_res_m, rotation_deg)
        else:
            old_vec_x_row, old_vec_y_row = simple_rotate(0, old_res_m, rotation_deg)
            
        d_row = first['row'] - min_row
        d_col = first['col'] - min_col
        
        origin_utm_x = fx - (d_col * old_vec_x_col + d_row * old_vec_x_row)
        origin_utm_y = fy - (d_col * old_vec_y_col + d_row * old_vec_y_row)

    # 5. Calculate New Basis Vectors (Refined)
    def rotate_point(x, y, angle_deg):
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_rot = x * cos_a + y * sin_a
        y_rot = -x * sin_a + y * cos_a
        return x_rot, y_rot

    # Col vector moves East-ish
    col_vec_x, col_vec_y = rotate_point(new_res_m, 0, rotation_deg)
    
    # Row vector moves South (geospatial) or North (cartesian)
    if convention == 'geospatial':
        row_vec_x, row_vec_y = rotate_point(0, -new_res_m, rotation_deg)
    else:
        row_vec_x, row_vec_y = rotate_point(0, new_res_m, rotation_deg)

    # 6. Generate New Points
    new_points_data = []
    
    # We iterate 0 to 2N-1. 
    # 0 maps to origin. 
    # 1 maps to origin + 0.5*old_res (midpoint).
    # 2N-1 maps to origin + (2N-1)*new_res = origin + (N - 0.5)*old_res (half step past old end).
    for r in range(n_rows_new):
        row_offset_x = r * row_vec_x
        row_offset_y = r * row_vec_y
        
        for c in range(n_cols_new):
            col_offset_x = c * col_vec_x
            col_offset_y = c * col_vec_y
            
            # Calculate UTM position
            p_utm_x = origin_utm_x + row_offset_x + col_offset_x
            p_utm_y = origin_utm_y + row_offset_y + col_offset_y
            
            # Transform to output CRS
            out_x, out_y = transformer_to_output.transform(p_utm_x, p_utm_y)
            
            # Use 0-based indexing for the fresh high-res grid
            new_points_data.append({
                'geometry': Point(out_x, out_y),
                'col': c,
                'row': r,
                'convention': convention,
                'res_km': new_res_km,
                'rot_deg': rotation_deg,
                'utm_crs': local_utm
            })

    return gpd.GeoDataFrame(new_points_data, crs=output_crs)

def create_grid_boundary_polygon(gdf_grid):
    """
    Create a polygon that connects the outer boundary points of a grid.
    """
    if len(gdf_grid) == 0:
        raise ValueError("Input grid is empty")
    
    min_col = gdf_grid['col'].min()
    max_col = gdf_grid['col'].max()
    min_row = gdf_grid['row'].min()
    max_row = gdf_grid['row'].max()
    
    boundary_points = []
    
    # Top edge
    top_edge = gdf_grid[gdf_grid['row'] == min_row].sort_values('col')
    for _, point in top_edge.iterrows():
        boundary_points.append((point.geometry.x, point.geometry.y))
    
    # Right edge
    right_edge = gdf_grid[gdf_grid['col'] == max_col].sort_values('row')
    for _, point in right_edge.iloc[1:].iterrows():
        boundary_points.append((point.geometry.x, point.geometry.y))
    
    # Bottom edge
    bottom_edge = gdf_grid[gdf_grid['row'] == max_row].sort_values('col', ascending=False)
    for _, point in bottom_edge.iloc[1:].iterrows():
        boundary_points.append((point.geometry.x, point.geometry.y))
    
    # Left edge
    left_edge = gdf_grid[gdf_grid['col'] == min_col].sort_values('row', ascending=False)
    for _, point in left_edge.iloc[1:-1].iterrows():
        boundary_points.append((point.geometry.x, point.geometry.y))
    
    polygon = Polygon(boundary_points)
    
    gdf_polygon = gpd.GeoDataFrame({
        'geometry': [polygon],
        'convention': [gdf_grid['convention'].iloc[0]],
        'res_km': [gdf_grid['res_km'].iloc[0]],
        'rot_deg': [gdf_grid['rot_deg'].iloc[0]],
        'grid_cols': [max_col - min_col + 1],
        'grid_rows': [max_row - min_row + 1]
    }, crs=gdf_grid.crs)
    
    return gdf_polygon


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import json
    import numpy as np

    # Configuration paths
    JSON_FILE = os.path.join('GIS', 'regions', 'regions_meta.json')
    INPUT_CRS = 'EPSG:4326'
    OUTPUT_CRS = 'EPSG:4326'

    # 1. Load Regions from JSON
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found. Please check the path.")
        exit(1)

    with open(JSON_FILE, 'r') as f:
        regions_data = json.load(f)

    print(f"Found {len(regions_data)} regions to process.")
    
    for i, region in enumerate(regions_data, start=1):
        name = region.get('name', f"region_{i:02d}")
        
        folder_name = f"region_{i:02d}"
        output_dir = os.path.join('GIS', 'regions', folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"PROCESSING: {name} -> Saving to {output_dir}/")
        print("="*70)
        
        # ---------------------------------------------------------
        # A. Create Base Grid
        # ---------------------------------------------------------
        print(f"Creating base grid with rotation {region['rot_deg']} deg...")
        gdf_grid = create_region_grid(
            input_point=region['point'],
            input_crs=INPUT_CRS,
            point_type=region['point_type'],
            grid_size=region['core_dims'], 
            resolution_km=region['res_km'],
            output_crs=OUTPUT_CRS,
            rotation_deg=region['rot_deg']
        )

        # ---------------------------------------------------------
        # B. Create LR Grid
        # ---------------------------------------------------------
        lr_dil = region['lrgrid_dilation']
        gdf_lr = dilate_grid(
            gdf_grid, 
            row_north=lr_dil['north'], 
            row_south=lr_dil['south'],
            col_west=lr_dil['west'], 
            col_east=lr_dil['east']
        )
        
        # Create MR Grid (Refine LR once)
        gdf_mr = refine_grid_2x(gdf_lr)

        # ---------------------------------------------------------
        # C. Create HR Grid (Dilated + Refined 4x)
        # ---------------------------------------------------------
        hr_dil = region['hrgrid_dilation']
        
        # 1. Dilate base grid
        gdf_hr_temp = dilate_grid(
            gdf_grid, 
            row_north=hr_dil['north'], 
            row_south=hr_dil['south'],
            col_west=hr_dil['west'], 
            col_east=hr_dil['east']
        )
        
        # 2. Refine 4x (Apply 2x refinement twice)
        gdf_hr_2x = refine_grid_2x(gdf_hr_temp)
        gdf_hr = refine_grid_2x(gdf_hr_2x)

        # ---------------------------------------------------------
        # D. Enforce Major Dimension First (Reshape)
        # ---------------------------------------------------------
        def ensure_major_dim_first(gdf, label):
            """
            Checks if Cols > Rows. If so, swaps 'row' and 'col' columns
            so that the first dimension (Rows) is always the larger number.
            """
            n_rows = gdf['row'].max() + 1
            n_cols = gdf['col'].max() + 1
            
            if n_cols > n_rows:
                print(f"  > {label}: Reshaping ({n_rows}x{n_cols}) -> ({n_cols}x{n_rows}) to put larger dim first.")
                # Swap row and col values
                r_vals = gdf['row'].copy()
                c_vals = gdf['col'].copy()
                gdf['row'] = c_vals
                gdf['col'] = r_vals
                return gdf, n_cols, n_rows # Return new shape
            
            return gdf, n_rows, n_cols

        # Apply to LR
        gdf_lr, lr_rows, lr_cols = ensure_major_dim_first(gdf_lr, "LR Grid")
        print(f"Creating Low-Res (LR) grid... Final Shape: ({lr_rows}, {lr_cols})")
        
        # Apply to MR
        gdf_mr, mr_rows, mr_cols = ensure_major_dim_first(gdf_mr, "MR Grid")
        print(f"Creating Medium-Res (MR) grid... Final Shape: ({mr_rows}, {mr_cols})")

        # Apply to HR
        gdf_hr, hr_rows, hr_cols = ensure_major_dim_first(gdf_hr, "HR Grid")
        print(f"Creating High-Res (HR) grid... Final Shape: ({hr_rows}, {hr_cols})")

        # ---------------------------------------------------------
        # E. Save Shapefiles (with updated indices)
        # ---------------------------------------------------------
        file_lr = os.path.join(output_dir, f'{name}_lr_grid.shp')
        gdf_lr.to_file(file_lr)
        print(f"  > Saved Grid SHP: {file_lr}")

        file_mr = os.path.join(output_dir, f'{name}_mr_grid.shp')
        gdf_mr.to_file(file_mr)
        print(f"  > Saved Grid SHP: {file_mr}")

        file_hr = os.path.join(output_dir, f'{name}_hr_grid.shp')
        gdf_hr.to_file(file_hr)
        print(f"  > Saved Grid SHP: {file_hr}")

        # ---------------------------------------------------------
        # F. Save Arrays (Lat/Lon)
        # ---------------------------------------------------------
        def save_latlon_array(gdf, n_rows, n_cols, out_path):
            # Sort to ensure reshaping maps correctly to (row, col) indices
            df_pivot = gdf.sort_values(by=['row', 'col'])
            
            # Extract and reshape
            lons_array = df_pivot.geometry.x.values.reshape(n_rows, n_cols)
            lats_array = df_pivot.geometry.y.values.reshape(n_rows, n_cols)
            
            np.savez_compressed(out_path, lats=lats_array, lons=lons_array)

        # Save LR Array
        file_lr_npz = os.path.join(output_dir, f'{name}_lr_latlon.npz')
        save_latlon_array(gdf_lr, lr_rows, lr_cols, file_lr_npz)
        print(f"  > Saved Array: {file_lr_npz} {lr_rows}x{lr_cols}")

        # Save MR Array
        file_mr_npz = os.path.join(output_dir, f'{name}_mr_latlon.npz')
        save_latlon_array(gdf_mr, mr_rows, mr_cols, file_mr_npz)
        print(f"  > Saved Array: {file_mr_npz} {mr_rows}x{mr_cols}")

        # Save HR Array
        file_hr_npz = os.path.join(output_dir, f'{name}_hr_latlon.npz')
        save_latlon_array(gdf_hr, hr_rows, hr_cols, file_hr_npz)
        print(f"  > Saved Array: {file_hr_npz} {hr_rows}x{hr_cols}")

        # ---------------------------------------------------------
        # G. Save Boundary Polygons
        # ---------------------------------------------------------
        # Base Boundary
        gdf_bound_base = create_grid_boundary_polygon(gdf_grid)
        file_bound_tg = os.path.join(output_dir, f'{name}_tidegauge_boundary.shp')
        gdf_bound_base.to_file(file_bound_tg)
        print(f"  > Saved Boundary: {file_bound_tg}")

        # LR Boundary
        gdf_bound_lr = create_grid_boundary_polygon(gdf_lr)
        file_bound_lr = os.path.join(output_dir, f'{name}_lr_boundary.shp')
        gdf_bound_lr.to_file(file_bound_lr)
        print(f"  > Saved Boundary: {file_bound_lr}")

        # HR Boundary
        gdf_bound_hr = create_grid_boundary_polygon(gdf_hr)
        file_bound_hr = os.path.join(output_dir, f'{name}_hr_boundary.shp')
        gdf_bound_hr.to_file(file_bound_hr)
        print(f"  > Saved Boundary: {file_bound_hr}")