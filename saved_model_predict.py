from prac1 import preprocess_data,load_model,imgtoarray,modeleval,showpred

train,test,sample_submission=preprocess_data()

model = load_model('prac2.h5')

test_x=imgtoarray(test,dir_name='test')

pred=modeleval(model,test_x)

showpred(test,train,pred)