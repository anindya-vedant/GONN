import pickle
infile=open('finalized_model_1.pkl','rb')
new_dict=pickle.load(infile)
infile.close()

print(new_dict)
