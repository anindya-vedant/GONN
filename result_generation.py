#import final_code
import importlib
for i in range(10):
	import final_code
	import new_gp_arpit_sir
	importlib.reload(new_gp_arpit_sir)
	print("Result Number: ", i+1)