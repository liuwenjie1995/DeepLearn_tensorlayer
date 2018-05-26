import tensorlayer as tl
def load_fashion_minist(type):
    if type==1:
        shape=(-1,784)
    elif type==2:
        shape=(-1,28,28,1)
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_fashion_mnist_dataset(shape=shape,path="E:\Work\deep_learn_data")
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_voc(time):
    imgs_file_list, imgs_semseg_file_list, imgs_insseg_file_list, imgs_ann_file_list,classes, classes_in_person, classes_dict,n_objs_list, objs_info_list, objs_info_dicts = tl.files.load_voc_dataset(dataset=str(time), contain_classes_in_person=False,path="E:\Work\deep_learn_data")
    return  imgs_file_list, imgs_semseg_file_list, imgs_insseg_file_list