import gpt_2_simple as gpt2
import tensorflow as tf
import json
from flask import Flask

#from tensorflow.python.eager.monitoring import Buckets
#import boto3

app = Flask()

'''
s3 = boto3.client('s3')
bucket = 'paperplane-ai'
checkpoint = s3.get_object(Bucket=bucket, Key='checkpoint/*')
'''

tf.compat.v1.reset_default_graph()
sess = gpt2.start_tf_sess()    
gpt2.load_gpt2(sess, run_name='run1')

@app.route("/")
def shrink(i_context):

    p = (gpt2.generate(sess,
                    length=300,
                    temperature=0.75,
                    top_p=0.8,
                    prefix=i_context,
                    top_k=40,
                    nsamples=5,
                    batch_size=5,
                    return_as_list=True 
                    ))
    
    q = ''
    for _ in p:
        _ = ".".join(_.split(".")[:-1])        
        _ = _.replace('\n','')
        _ = _.replace('\\','')
        
        q = q+_+'##'
        
    q = q.split('##')    
    return q

