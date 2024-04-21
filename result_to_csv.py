import numpy as np
import pandas as pd


def get_result(r):
    result = np.zeros((len(r['presion']), len(r)))
    result[:, 0] = r['accuracy']
    for i in range(len(r['presion'])):
        result[i, 1] = r['presion'][i]
        result[i, 2] = r['recall'][i]
        result[i, 3] = r['F1score'][i]
        result[i, 4] = r['Iou'][i]
    result[:, 5] = r['FWIou']
    result[:, 6] = r['mIou']
    result[:, 7] = r['loss']
    return result
    
def result_to_csv(result_list, out_file='every_epoch_result.csv'):
    """
    return a csv file with the result of every epoch
    
    example:
    >>>
        accuracy    precision	 recall	    F1score     Iou	        FWIou	    mIou        loss
    1	0.869576	0.941980	0.878577	0.909174	0.833474	0.779735	0.728924	0.374791
    1	0.869576	0.706144	0.843553	0.768756	0.624374	0.779735	0.728924	0.374791
    2	0.915612	0.958452	0.924376	0.941106	0.888763	0.848764	0.814854	0.278165
    2	0.915612	0.813977	0.891987	0.851198	0.740945	0.848764	0.814854	0.278165
    ....

    """
    # result = np.array(result_list)
    result = get_result(result_list[0])
    for i in range(1, len(result_list)):
        result = np.vstack((result, get_result(result_list[i])))
        
    name = ['accuracy','precision','recall','F1score','Iou','FWIou','mIou','loss']
    df2 = pd.DataFrame((result))
    df2.index = sorted(list(range(1, len(result_list)+1))*2)
    df2.columns = name
    df2.to_csv(out_file, index=True, encoding="utf_8_sig")
    print('save result to {}'.format(out_file))
    return df2