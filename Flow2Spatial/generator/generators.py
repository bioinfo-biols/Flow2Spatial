from .design_utils import *
def omics(adata, mask, dir_run='./save_environ'):
    #based on reference spatial omics
    #input:
    #   adata: spatial distribution of molecular from reference omics
    #   mask: bool matrix, showing whether tissue slice is palced in this pixel
    #   dir_run: directory for saving spatial omics data
    #

    adata_xa = adata.X.A
    point_row, point_col = np.where(mask)

    out_list_transfer = []
    for n in range(adata.X.shape[1]):
        tmp = transfer_mask_rw(adata_xa[:,1], mask, point_row, point_col)
        out_list_transfer.append(tmp)

    if not os.path.exists(dir_run):
        os.makedirs(dir_run)

    with open(str(dir_run) + '/omics', 'wb') as handle:
        pickle.dump(np.array(out_list_transfer), handle)

    return('Omics data done')


def histology(line_row, line_col, mask, segments, channel_intensity, radius=[0.5, 0.5], dir_run='./save_environ'):
    #based on segments by histological information(including: spatial clusters)
    #input
    #segments: list

    point_row, point_col = np.where(mask)
    max_rc = max(max(point_row), max(point_col))

    low_row, low_col = radius
    one_size = 4 * low_row * low_col

    design_row = []
    for i in range(1, len(line_row)):
        pol1_rc = traverse_generate_vertex(line_row['line'+str(i)], line_row['line'+str(i+1)], max_rc)
        tmp = traverse_poly_search_radis(point_row, point_col, pol1_rc, x_radis=low_row, y_radis=low_col)
        design_row.append(transfer_mask_rw(tmp/one_size, mask, point_row, point_col))
        
        
    design_col = []
    for i in range(1, len(line_col)):
        pol1_rc = traverse_generate_vertex(line_col['line'+str(i)], line_col['line'+str(i+1)], max_rc)
        tmp = traverse_poly_search_radis(point_row, point_col, pol1_rc, x_radis=low_row, y_radis=low_col)
        design_col.append(transfer_mask_rw(tmp/one_size, mask, point_row, point_col))
        
    if not os.path.exists(dir_run):
        os.makedirs(dir_run)
        
    with open(dir_run + '/design_row', 'wb') as handle:
        pickle.dump(design_row, handle)

    with open(dir_run + '/design_col', 'wb') as handle:
        pickle.dump(design_col, handle)

    with open(dir_run + '/mask', 'wb') as handle:
        pickle.dump(mask, handle)

    ## segments design_row design_col
    channel_count_dat_list = iterative_channel_count(segments, design_x=design_row, design_y=design_col, mask=mask)

    y0 = [(mask.flatten().astype(int) * tmp).sum() for tmp in design_row]
    y90 = [(mask.flatten().astype(int) * tmp).sum() for tmp in design_col]
    length_coef = [y0, y90]
    with open(dir_run + '/length_coef', 'wb') as handle:
        pickle.dump(length_coef, handle)

    ## computation
    from .utils import *
    ### will be accelerated by mulitiprocess in later version
    segments_len = len(segments)
    for seg_i in range(segments_len):
        similar_generator(seg_i, segments, channel_count_dat_list, design_row, design_col, channel_intensity, mask, length_coef, dir_run)
        print(seg_i)

    ## read parallel results
    read_out_list = []
    for seg_i in range(segments_len):#range(1,4):
        with open(str(dir_run) + '/histology' + str(seg_i) , 'rb') as handle:
            tmp = pickle.load(handle)
            read_out_list.extend(tmp)

    out_list_array_similar = np.array(read_out_list)
    with open(dir_run + '/histology', 'wb') as handle:#
        pickle.dump(out_list_array_similar, handle)

    return('Histological data done')


def random(input_type=['omics', 'histology'], dir_run='./save_environ', times=20000):
    #based on randomness
    ##random

    if (len(input_type) == 2):
        with open(dir_run + '/histology', 'rb') as handle:
            out_list_array_similar = pickle.load(handle)

        with open(dir_run + '/omics', 'rb') as handle:
            out_list_array_transfer = pickle.load(handle)
        
    else:
        if input_type == ['omics']:
            with open(dir_run + '/omics', 'rb') as handle:
                out_list_array_similar = pickle.load(handle)
                out_list_array_transfer =  out_list_array_similar

        elif input_type == ['histology']:
            with open(dir_run + '/histology', 'rb') as handle:
                out_list_array_similar = pickle.load(handle)
                out_list_array_transfer =  out_list_array_similar            

    random_target_similar = []
    for m in range(times):
        a = out_list_array_similar[np.random.choice(out_list_array_similar.shape[0], 1, replace=False),:].flatten()
        b = out_list_array_transfer[np.random.choice(out_list_array_transfer.shape[0], 1, replace=False),:].flatten()
        a = a / (np.mean(a)+1e-3)
        b = b / (np.mean(b)+1e-3)
        if m % 7 == 0:
            random_target = a - b
        else:
            random_target = a + b

        random_target[random_target<0] = 0
        
        random_target_similar.append(random_target)

    random_target_similar_array = np.array(random_target_similar)
    with open(dir + '/random', 'wb') as handle:#
        pickle.dump(random_target_similar_array, handle)

    return('Random data done')



