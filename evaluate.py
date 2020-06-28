import tensorflow as tf
import pathlib
from model.dataset import decode_img
from model.hardTripletLoss import _pairwise_distances

# function to pre-process test image samples 
def get_images(path):
    img = tf.io.read_file(path)
    img = decode_img(img)
    return img

# function to evaluate the performance of the model based on the recall,
# precision, accuracy, f1-score, far, frr and eer performance metrics
def evaluate_model(TP,TN,FP,FN):
    '''
    TP: no. of true positives
    TN: no. of true negatives
    FP: no. of false positives
    TN: no. of false negatives
    '''
    recall = TP / (TP + FN)         # determine the recall        
    precision = TP / (TP + FP)      #determine the precision
    f1_score = 2.0 * (recall * precision) / (recall + precision)  # determine the f1-score      
    far = FP / (FP + TN)            # determine the false acceptance rate        
    frr = FN / (FN + TP)            # determine the false rejection rate        
    eer = (far + frr) / 2.0         # determine the equal error rate        
    acc = 1.0 - eer                 # lastly the accuracy
    return {        
        "recall" : recall,       
        "precision" : precision,         
        "f1-score" : f1_score,     
        "far" : far,        
        "frr" : frr,        
        "eer" : eer,        
        "acc" : acc,
    }

if __name__ == "__main__":
    # Load the model for testing
    mossnet = tf.keras.models.load_model("saved_model/MOSSNET", compile=False)

    try:
        # Load images for testing the model
        ref_dir = pathlib.Path('../input/mossignatures/TestSet/References')
        pos_dir = pathlib.Path('../input/mossignatures/TestSet/Queries/genuine')
        neg_dir = pathlib.Path('../input/mossignatures/TestSet/Queries/forged')
        
        ref_gen = ref_dir.glob('**/*1.png')     # load reference signature samples
        refs = [str(i) for i in ref_gen]        
        references = map(get_images, refs) 

        pos_gen = pos_dir.glob('**/*.png')      # load origianl signature samples
        pos = [str(i) for i in pos_gen]         
        p_queries = map(get_images, pos)

        neg_gen = neg_dir.glob('**/*.png')      # load counterfeit signature samples
        negs = [str(i) for i in neg_gen]       
        n_queries = map(get_images, negs)       


        r = [i for i in references]
        p = [i for i in p_queries]
        n = [i for i in n_queries]

        # define threshhold for differenciating counterfeits from originals
        threshold = tf.constant([1.16])     # Ideal is 1.0 at 0 loss. 

        #Test model on oiginal signature samples
        TP, FN = 0, 0  # TP: True positives FN: False negatives

        for i, reference in enumerate(r):
            reference = tf.reshape(reference, shape=[1, 128, 128, 3])
            anchor = mossnet(reference) # obtain feature embedding of reference signature
            index = [i*4, i*4+1, i*4+2, i*4+3] 
            for query in tf.gather(p, index):
                query = tf.reshape(query, shape=[1, 128, 128, 3])
                test = mossnet(query)   # obtain feature embedding of the queried sample
                # determine the pairwise distance between the reference and query embeddings
                pdist_matrix = _pairwise_distances(tf.concat([anchor,test],0))
                pairwise_distance = tf.slice(pdist_matrix, [0,1], [1,1])
                pairwise_distance = tf.reshape(pairwise_distance, [1])
                # compare pairwise distance between them to the threshold
                if tf.math.less_equal(pairwise_distance,threshold):
                    TP += 1 # predict query as a original if the distance is not greater than 1.0
                else:
                    FN += 1 # predict query as counterfeit otherwise
            
        # Test model on counterfeit signature samples
        TN, FP = 0, 0 # TN: True negatives  FP: False positives 

        for i, reference in enumerate(r):
            reference = tf.reshape(reference, shape=[1, 128, 128, 3])
            anchor = mossnet(reference) # obtain feature embedding of reference signature
            index = [i*4, i*4+1, i*4+2, i*4+3]
            for query in tf.gather(n, index):
                query = tf.reshape(query, shape=[1, 128, 128, 3])
                test = mossnet(query)   # obtain feature embedding of the queried sample
                # determine the pairwise distance between the reference and query embeddings
                pdist_matrix = _pairwise_distances(tf.concat([anchor,test],0))
                pairwise_distance = tf.slice(pdist_matrix, [0,1], [1,1])
                pairwise_distance = tf.reshape(pairwise_distance, [1])
                # compare pairwise distance between them to the threshold
                if tf.math.greater(pairwise_distance,threshold): 
                    TN += 1 # predict query as a forgery if the distance is greater than 1.0
                else:
                    FP += 1 # predict query as original otherwise

        # Evaluate model
        performance = evaluate_model(TP, TN, FP, FN)
        print(performance)

    except Exception as exp:
        print(f'No model has been saved yet \n {exp}')



