# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

def get_optuna_params(model_type: str, dataset: str) -> dict:
    params = {}
    if dataset=='CIFAR10':
        if model_type=='resnet20':
            params["lr"] = 0.0016822249163093617
            params["lr_bias"] = 63.934695046801245
            params["momentum"] =  0.8484574950771097
            params["weight_decay"] = 0.11450934135118791
        elif model_type=='resnet32':
            # 91.19: lr: 0.0013205254360784781, lr_bias: 61.138281101282544, momentum: 0.873508553678625, weight_decay: 0.26911634559915815
            # 91.1 : lr: 0.0013978655308274968, lr_bias: 70.43940111170473,  momentum: 0.8611100787383372, weight_decay: 0.2604742590264777
            # 90.99: lr: 0.0019695910893940986, lr_bias: 60.930501987151686, momentum: 0.8831260271578129, weight_decay: 0.1456126229025426
            params["lr"] = 0.0013205254360784781
            params["lr_bias"] = 61.138281101282544
            params["momentum"] = 0.873508553678625
            params["weight_decay"] = 0.26911634559915815
        elif model_type=='resnet44':
            # 91.49: 0.0017177668853317557	72.4258603207131	0.8353896320183106	0.16749858871622
            # 91.16: 0.0019608745758959625	67.9132255882833	0.8041541468923449	0.19517278422517992
            # 91.01: 0.0009350979452929332	71.95838038824016	0.858379476548086	0.06780300392316674
            params["lr"] = 0.0017177668853317557
            params["lr_bias"] = 72.4258603207131
            params["momentum"] =  0.8353896320183106
            params["weight_decay"] = 0.16749858871622
        elif model_type=='resnet56':
            # 92.12: 0.0012022823706985977	71.31108702685964	0.8252747623136261	0.26463818739336625
            # 91.90: 0.0010850336892236205	55.20534833175523	0.8738224946147084	0.10705317777179325
            # 91.56: 0.0019151327847040805	63.38376732305882	0.9134938189630787	0.24446065595718675
            params["lr"] = 0.0012022823706985977
            params["lr_bias"] = 71.31108702685964
            params["momentum"] =  0.8252747623136261
            params["weight_decay"] = 0.26463818739336625
        elif model_type=='resnet110':
            # 92.23: 0.001477698037686629	61.444988882569774	0.7241645867415002	0.23586225065185779
            # 92.17: 0.0017110807237653582	65.2511959805971	0.8078620231092996	0.19065715813207001
            # 92.16: 0.0015513227695282382	59.89497310126697	0.7355843250067341	0.13248840913478463
            params["lr"] = 0.001477698037686629
            params["lr_bias"] = 61.444988882569774
            params["momentum"] =  0.7241645867415002
            params["weight_decay"] = 0.23586225065185779
        else:
            print("model_type and dataset are incorrectly specified. Returning resnet20 params.")
            params["lr"] = 0.0016822249163093617
            params["lr_bias"] = 63.934695046801245
            params["momentum"] =  0.8484574950771097
            params["weight_decay"] = 0.11450934135118791
    else:
        # only return resnet32 since CIFAR100 only needs this
        if model_type=='resnet32':
            # 65.09: 0.0018636209167742187	64.96657354785438	0.9186032548289501	0.15017464467868924
            # 64.70: 0.0017509006966116355	60.10884856596049	0.8921508582343675	0.10919043636429121
            # 64.49: 0.0015358614659175514	59.175398449172015	0.8553794786037812	0.20824545084283141
            params["lr"] = 0.0018636209167742187
            params["lr_bias"] = 64.96657354785438
            params["momentum"] =  0.9186032548289501
            params["weight_decay"] = 0.15017464467868924
        else:
            # Default bag of trick params
            params["lr"] = 0.001
            params["lr_bias"] = 64
            params["momentum"] =  0.9
            params["weight_decay"] = 0.256

    print("Loading params for %s, %s" % (model_type, dataset))
    print(params)
    return params
