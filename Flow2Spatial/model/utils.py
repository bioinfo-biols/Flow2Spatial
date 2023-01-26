from .dnn_utils import *
import pickle

def slicing(dataset_array, design_row, design_col):
    ## slicing data

    # dataset_array = target_array
    # design_row = design_row
    # design_col = design_col

    out_x = []
    out_y = []

    ##only considering in len(design_row) == len(design_col)
    for design_i in range(min(len(design_row), len(design_col))):
        
        design_row_i = design_row[design_i]
        design_row_i_bool = design_row_i>0
        design_row_i_weight = design_row_i[design_row_i_bool]


        design_col_i = design_col[design_i]
        design_col_i_bool = design_col_i>0
        design_col_i_weight = design_col_i[design_col_i_bool]

        tmp_x = []
        tmp_y = []
        for gene_i in range(dataset_array.shape[0]):

            gene_exp = dataset_array[gene_i,:]
            intensity_x = (gene_exp[design_row_i_bool] * design_row_i_weight).sum()
            
            intensity_y = (gene_exp[design_col_i_bool] * design_col_i_weight).sum()
            tmp_y.append(intensity_y)
                
            tmp_x.append(intensity_x)
            

        out_x.append(tmp_x)
        out_y.append(tmp_y)

    if len(design_row) > len(design_col):
        for design_i in range(min(len(design_row), len(design_col)), len(design_row)):
            design_row_i = design_row[design_i]
            design_row_i_bool = design_row_i>0
            design_row_i_weight = design_row_i[design_row_i_bool]
            tmp_x = []
            tmp_y = []
            for gene_i in range(dataset_array.shape[0]):
                gene_exp = dataset_array[gene_i,:]
                intensity_x = (gene_exp[design_row_i_bool] * design_row_i_weight).sum()
                tmp_x.append(intensity_x)

            out_x.append(tmp_x)
        
    if len(design_row) < len(design_col):
        for design_i in range(min(len(design_row), len(design_col)), len(design_col)):
            design_col_i = design_col[design_i]
            design_col_i_bool = design_col_i>0
            design_col_i_weight = design_col_i[design_col_i_bool]
            tmp_x = []
            tmp_y = []
            for gene_i in range(dataset_array.shape[0]):
                gene_exp = dataset_array[gene_i,:]
                intensity_y = (gene_exp[design_col_i_bool] * design_col_i_weight).sum()
                tmp_y.append(intensity_y)
                
            out_y.append(tmp_y)

    out_row_T = np.array(out_x).T#.shape
    out_col_T = np.array(out_y).T#.shape

    input_instensity = np.hstack([out_row_T, out_col_T])
    print(input_instensity.shape)

    return(input_instensity)

def format_dnn(i, target_array, input_instensity, file_gene_id, design_row):
    assess_id = file_gene_id[i]
    
    intensity = input_instensity[i,:]

    half_tmpX = intensity[0:len(design_row)][::2]
    half_tmpY = intensity[len(design_row):][::2]
    
    half_tmpXY_mean = (1e-5+np.mean(np.concatenate([half_tmpX, half_tmpY]))/500)
    half_tmpX /= half_tmpXY_mean
    half_tmpY /= half_tmpXY_mean
    
    Xrep = np.tile(np.array(half_tmpX)[:,np.newaxis], int(len(half_tmpY)))
    Yrep = np.array([half_tmpY] * int(len(half_tmpX)))
    XYrep = Xrep + Yrep
    
    input_XYrep = np.float32(np.stack((Xrep, Yrep, XYrep), axis=0))
    np.save('./DNN_data/data/'+ assess_id + '.npy', input_XYrep)
    
    random_target = target_array[i,:]
    random_Target_norm = random_target / half_tmpXY_mean

    return([assess_id] + list(random_Target_norm))
    
def preparation(input_type=['omics', 'histology', 'random'], dir_run='./save_environ', testing=0.1, mask='mask', design_row_file='design_row', design_col_file='design_col'):

    input_array = []
    for filei in input_type:
        with open(dir + '/' + str(filei), 'rb') as handle:
            tmp = pickle.load(handle)
            input_array.append(tmp)

    target_array = np.vstack(input_array)
    target_array[target_array < 0] = 0

    with open(dir_run + '/' + design_row_file, 'rb') as handle:
        design_row = pickle.load(handle)

    with open(dir_run + '/' + design_col_file, 'rb') as handle:
        design_col = pickle.load(handle)

    input_instensity = slicing(target_array, design_row, design_col)

    ## generate format for deep learning
    file_gene_id = ['gene' + str(i) for i in range(target_array.shape[0])]

    if not os.path.exists(dir_run + '/DNN_data'):
        os.makedirs(dir_run + '/DNN_data')
    if not os.path.exists(dir_run + '/DNN_data/data'):
        os.makedirs(dir_run + '/DNN_data/data')
        
    _a = (format_dnn(i, target_array, input_instensity, file_gene_id, design_row) for i in range(target_array.shape[0]))
    random_target_save_dat = pd.DataFrame(_a)

    random_target_save_dat_dropna = random_target_save_dat.dropna()

    combined_target_save_dat = random_target_save_dat_dropna
    tmp = 1 - testing
    combined_target_save_dat_part_90 = combined_target_save_dat.sample(frac = tmp)

    # Creating dataframe with
    # rest of the 10% values
    rest_part_10 = combined_target_save_dat.drop(combined_target_save_dat_part_90.index)

    with open(dir_run + '/DNN_data/training_data', 'wb') as handle:
        pickle.dump(combined_target_save_dat_part_90, handle)

    with open(dir_run + '/DNN_data/testing_data', 'wb') as handle:
        pickle.dump(rest_part_10, handle)
    
    with open(dir_run + '/' + mask, 'rb') as handle:
        mask = pickle.load(handle)

    with open(dir_run + '/DNN_data/mask', 'wb') as handle:
        pickle.dump(mask, handle)

    return('Data preparation done')


def training(DNN_para=[12, 10, 8], batch_size = 32, learning_rate=1e-3, epochs = 100, save_epoch=2, y_flag = 0, dir_run='./save_environ'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    with open(dir_run + '/DNN_data/training_data', 'rb') as handle:
        combined_target_save_dat_part_90 = pickle.load(handle)

    with open(dir_run + '/DNN_data/testing_data', 'rb') as handle:
        rest_part_10 = pickle.load(handle)
        
    training_data = MyDataset(combined_target_save_dat_part_90, dir_run + '/DNN_data/data/', device)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    test_data = MyDataset(rest_part_10, dir_run + '/DNN_data/data/', device)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    with open(dir_run + '/DNN_data/mask', 'rb') as handle:
        mask = pickle.load(handle)

    mask_model_T = torch.tensor(mask)
    mask_model_T = mask_model_T.to(device)

    model = NbcNet(DNN_para)
    model.eval()
    model = model.to(device)

    # learning_rate = 1e-3
    # epochs = 200
    model_dir = dir_run + '/DNN_data/models'
    # y_flag = 0

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    test_loss = []
    for t in range(epochs):
        loss = train_loop(train_dataloader, model, loss_fn, optimizer, mask_model_T, device, y_flag)
        train_loss.append(loss)
        
        tloss = test_loop(test_dataloader, model, loss_fn, mask_model_T, device, y_flag)
        test_loss.append(tloss)
        
        if (t) % save_epoch == 0:
            print(f"Epoch {t}\n-------------------------------")
            print(loss)
            print(tloss)
            
        if (t+1) % save_epoch == 0:
            torch.save(model.state_dict(), "./%s/Recontruct_weights_%d.pkl" % (model_dir, t+1))  # save only the parameters
        
    loss_dat = pd.DataFrame({'train_loss':train_loss, 'test_loss':test_loss})
    loss_dat['Epoch'] = loss_dat.index + 1
    loss_dat.to_csv(dir_run + '/loss.csv')

    print('Training Done')

def make_uniq_list(a):
    return [ a[i] + '_' + str(a[0:i].count(a[i])+1) if a[0:i].count(a[i]) > 0 else a[i] for i in range(len(a))]

def reconstruction(select_epoch, channel_intensity, out_adata='adata', DNN_para=[12, 10, 8], dir_run='./save_environ'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = NbcNet(DNN_para)

    model.load_state_dict(torch.load(dir_run + '/DNN_data/models/Recontruct_weights_' + str(select_epoch) + '.pkl'))

    model.eval()
    model = model.to(device)

    channel_intensity_list_dat = channel_intensity
    channel_intensity_list_dat.loc[channel_intensity_list_dat['PG.Genes'].isna(), 'PG.Genes'] = channel_intensity_list_dat.loc[channel_intensity_list_dat['PG.Genes'].isna(), 'PG.ProteinAccessions'] + '_gene'
    channel_intensity_list_dat_dropna = channel_intensity_list_dat.dropna()

    with open(dir_run + '/DNN_data/mask', 'rb') as handle:
        mask = pickle.load(handle)

    mask_model_T = torch.tensor(mask)
    mask_model_T = mask_model_T.to(device)

    with open(dir_run + '/design_x', 'rb') as handle:
        FDesign0 = pickle.load(handle)

    tmp_dat = {}
    channel_intensity_run_dat = channel_intensity_list_dat_dropna
    for i in range(channel_intensity_run_dat.shape[0]):
        tmp = generate_val_distribution(model, channel_intensity_run_dat.iloc[i,:], [2, (2+int(len(FDesign0)/2) + 1)], [(2+int(len(FDesign0)/2) + 1), channel_intensity_run_dat.shape[1]], mask_model_T, device)
        assess = channel_intensity_run_dat.index[i]
        
        tmp_dat[assess] = tmp.flatten()#[mask]

    tmp_dat_matrix = pd.DataFrame(tmp_dat).T

    tmp_dat_matrix.index = channel_intensity_list_dat_dropna['PG.ProteinAccessions']
    tmp_dat_matrix['Genes'] = channel_intensity_list_dat_dropna['PG.Genes'].values

    from anndata import AnnData
    import scanpy as sc

    Reconstruct_data = tmp_dat_matrix

    XX,YY = np.meshgrid(np.arange(mask.shape[1]),np.arange(mask.shape[0]))
    coord = np.vstack((mask.flatten(), XX.flatten(),YY.flatten())).T

    Reconstruct_data['PG'] = Reconstruct_data.index

    Reconstruct_data['Genes'] = [i[1:] if i.startswith(';') else i for i in Reconstruct_data['Genes']]
    Reconstruct_data['Genes'] = make_uniq_list(Reconstruct_data['Genes'].to_list())
    Reconstruct_data.index = Reconstruct_data['Genes']


    adata = AnnData(np.array(Reconstruct_data.iloc[:,np.append(mask[mask].flatten(), [False, False])].T), 
                    var = Reconstruct_data[['Genes', 'PG']],
                    obsm={"spatial": coord[coord[:,0]==1, 1:3]})
                    
    adata.write(dir_run + '/' + str(out_adata) + '.h5ad')#'/adata.h5ad'
    return('Reconstruction done')