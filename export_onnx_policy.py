import torch
import os 

# load the trained policy jit model 
policy_jit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs/policy_11.pt')
policy_jit_model = torch.jit.load(policy_jit_path)

#set the model to evalution mode
policy_jit_model.eval()

# creat a fake input to the model
test_input_tensor = torch.randn(1,77)  

#specify the path and name of the output onnx model
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('model_name', type=str, default='0')
# args = parser.parse_args()
model_name = "walk_run" #args.model_name
current_directory = os.getcwd()
policy_onnx_model = current_directory +f'/amp_{model_name}.onnx'

#export the onnx model
torch.onnx.export(policy_jit_model,               
                  test_input_tensor,       
                  policy_onnx_model,   # params below can be ignored
                  export_params=True,   
                  opset_version=11,     
                  do_constant_folding=True,  
                  input_names=['input'],    
                  output_names=['output'],  
                  )
# import onnx 
# model = onnx.load(policy_onnx_model)
# onnx.checker.check_model(model)
# print("ONNX 模型结构正确！")