import gpt_2_simple as gpt2
import tensorflow as tf
import json
from flask import Flask

#from tensorflow.python.eager.monitoring import Buckets
#import boto3

app = Flask(__name__)

'''
s3 = boto3.client('s3')
bucket = 'paperplane-ai'
checkpoint = s3.get_object(Bucket=bucket, Key='checkpoint/*')
'''

tf.compat.v1.reset_default_graph()
sess = gpt2.start_tf_sess()    
gpt2.load_gpt2(sess, run_name='run1')

@app.route("/shrink")
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

def lambda_handler(event, context):

    InputQuery = event['queryStringParameters']['msg']
    
    # warming up the lambda
    '''
    if event.get("source") in ["aws.events", "serverless-plugin-warmup"]:
        print('Lambda is warm!')
        return {}
    '''
    #body of response object
    res = {}
    res['message'] = shrink(InputQuery)
    
    # response object
    responseObject = {}
    responseObject['statusCode'] = 200
    responseObject['headers'] = {}
    responseObject['headers']['Content-type'] = 'application/json'
    responseObject['headers']['Access-Control-Allow-Origin'] = '*'
    responseObject['body'] = json.dumps(res)

    return responseObject
