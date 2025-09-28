from APQ.softwareSample import *
# LOAD THE APQ MODULES FIRST OTHERWISE THERE IS A CONFLICT OR TORCH WITH JOBLIB !!!
from _import_scripts import scripts
from scripts.notebook_utils import *
import csv
from scripts import *
import time

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.optimize import minimize
import math
import json
from scripts_new import *
import os
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.problem import ElementwiseProblem

from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

# Directory you want to create
EXTRA_COMPONENT = """
- !Component # Column readout (ADC)
  name: {}
  <<<: [*component_defaults, *keep_outputs, *no_coalesce]
  subclass: dummy_storage
  attributes: {{width: ENCODED_OUTPUT_BITS, n_bits: width, <<<: *cim_component_attributes}}
"""

# NeuroSim has some extra digital components. For fair comparison, we'll add as
# many components it has. We're not using them for energy so we'll just realize
# all of them as intadders.
EXTRA_NEUROSIM_COMPONENTS = ["shift_add", "adder", "pooling", "activation"]
EXTRA_COMPONENTS_CONFIG = "\n".join(
    EXTRA_COMPONENT.format(name) for name in EXTRA_NEUROSIM_COMPONENTS
)
countl=0
def run_layer(
    dnn: str,
    layer: str,
    input_bits,
    weights_bits,
    params: list,
    max_mappings: int = None,
):

    
    vl,bps,bl,csX,csY,srg,tic,mit,gbp=params
    vl,bl =[float(x) for x in [vl,bl]]
    bps,csX,csY,srg,tic,mit,gbp=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
    mis=1
    spec = get_spec(
        "basic_analog",
        tile="input_output_bufs",
        chip="large_router_glb",
        system="ws_dummy_buffer_many_macro",
        dnn=dnn,  
        layer=layer,
        jinja_parse_data={
            "cell_override": "rram_neurosim_default.cell.yaml",
            "ignoreme_placeholder": EXTRA_COMPONENTS_CONFIG,
        },
    )

    # NeuroSim's default macro variable settings
    spec.variables.update(
        dict(

            VOLTAGE=vl, 
            TECHNOLOGY=32,  # nm  --> adjust this
            BITS_PER_CELL=bps, 
            ADC_RESOLUTION=8, 
            VOLTAGE_DAC_RESOLUTION=1,

            N_SHIFT_ADDS_PER_BANK=16, 
            N_ADC_PER_BANK=16,
            BASE_LATENCY=bl, 
            READ_PULSE_WIDTH=1e-8,
            VOLTAGE_ENERGY_SCALE=1,
            VOLTAGE_LATENCY_SCALE=1,
            BATCH_SIZE=1,
        )
    )
    spec.architecture.find("row").spatial.meshY = csX  #vary this
    spec.architecture.find("column").spatial.meshX = csY #vary this
    spec.architecture.find("adc").attributes[
        "adc_estimator_plug_in"
    ] = '"Neurosim Plug-In"'

    
    output_bits=8
    spec.variables["WEIGHT_BITS"] = weights_bits
    spec.variables["INPUT_BITS"] = input_bits
    spec.variables["OUTPUT_BITS"] = output_bits
    spec.variables["TEMPORAL_DAC_RESOLUTION"] = input_bits
    
    spec.variables["MAX_UTILIZATION"] = False  # Do NOT generate a maximum-utilization workload; we're running a DNN workload.
    
    dt = spec.architecture.find("dummy_top").constraints.temporal
    
    dt.factors_only = dt.factors
    dt.factors.clear()
    dt.factors_only.add_eq_factor("X", input_bits)
    dt.factors_only.add_eq_factor("P", -1)
    dt.factors_only.add_eq_factor("Q", -1)
    spec.mapping.max_permutations_per_if_visit = 1

    dt = spec.architecture.find("glb").constraints.temporal
    dt.factors_only = dt.factors
    dt.factors.clear()
    dt.factors_only.add_eq_factor("X", input_bits)
    dt.factors_only.add_eq_factor("P", -1)
    dt.factors_only.add_eq_factor("Q", -1)
    spec.mapping.max_permutations_per_if_visit = 1

    #SYSTEM LVL
    spec.architecture.find("macro_in_system").spatial.meshX = mis 
    spec.architecture.find("macro_in_system").attributes.has_power_gating = True
    
    # ARCHITECTURE LVL
    spec.architecture.find("shared_router_group").spatial.meshX = srg
    spec.architecture.find("glb").attributes.depth = gbp 
    spec.architecture.find("shared_router_group").attributes.has_power_gating = True
    spec.architecture.find("tile_in_chip").spatial.meshX = tic 
    spec.architecture.find("tile_in_chip").attributes.has_power_gating = True

    #TILE LVL
    spec.architecture.find("macro_in_tile").spatial.meshX = mit 
    spec.architecture.find("macro_in_tile").attributes.has_power_gating = True

    # If there's a max_mappings, expand the search space
    if max_mappings is not None:
        # Set to evaluate max_mappings mappings
        spec.mapper.search_size = max_mappings
        spec.mapper.max_permutations_per_if_visit = max_mappings
        spec.mapper.victory_condition = max_mappings
        for t in ["row", "column", "macro", "dummy_top", "1bit_x_1bit_mac", "input_buffer", "output_buffer", "macro_in_tile", "glb", "router"]:
            t = spec.architecture.find(t)
            t.constraints.spatial.permutation.clear()
            t.constraints.temporal.permutation.clear()

    # Run the mapper
    return run_mapper(spec)


def getSpecs(dnn_name,layer_name):
    spec = get_spec(
        "basic_analog",
        tile="input_output_bufs",
        chip="large_router_glb",
        system="ws_dummy_buffer_many_macro",
        dnn=dnn_name,  # Set the DNN and layer
        layer=layer_name,
        jinja_parse_data={
            "cell_override": "rram_neurosim_default.cell.yaml",
            "ignoreme_placeholder": EXTRA_COMPONENTS_CONFIG,
        },
    )
    return spec

def divide_list_unequal(input_list, sizes):
    """
    Divides a list into unequal parts based on the given sizes.
    
    Parameters:
        input_list (list): The original list to be divided.
        sizes (list): A list of integers specifying the size of each part.
        
    Returns:
        list of lists: A list containing the divided parts.
    """
    if sum(sizes) > len(input_list):
        raise ValueError("The sum of sizes exceeds the length of the input list.")

    result = []
    start_index = 0
    
    for size in sizes:
        part = input_list[start_index:start_index + size]
        result.append(part)
        start_index += size
        
    return result



def getMetrics(paramsLST,dnn_name, num_mapping,n_jobs, layers, data_bits):
    
    DNN = dnn_name
    
    
    accuracy=data_bits['acc']
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(run_layer)(DNN, layer, data_bits[layer][0], data_bits[layer][1], paramsLST,num_mapping)
        for layer in layers
    )
    total_energy=0
    total_area=0
    total_latency=0
    total_mac=0
    sum_tops_per_mm2=0
    sum_tops_per_w=0
    sum_computes_per_second=0
    sum_computes_per_second_per_square_meter=0
    max_percent_utilization=0
    max_tops_per_w=0
    max_tops_per_mm2=0
    tops=0
    for r in [results]:
        for l in r:
            tops+=l.tops
            total_area=l.area*1e4 #in cm2 
            total_energy+=l.energy #in J
            total_latency+=l.latency #in seconds
            total_mac+=l.computes
            sum_tops_per_mm2+=l.tops_per_mm2
            sum_tops_per_w+=l.tops_per_w
            sum_computes_per_second+=l.computes_per_second
            sum_computes_per_second_per_square_meter+=l.computes_per_second_per_square_meter
            max_percent_utilization=max(max_percent_utilization, l.percent_utilization)
            max_tops_per_w=max(max_tops_per_w,l.tops_per_w)
            max_tops_per_mm2=max(max_tops_per_mm2, l.tops_per_mm2)
            
    avg_sum_tops_per_mm2=sum_tops_per_mm2/len(results)        
    avg_sum_tops_per_w=sum_tops_per_w/len(results)
    avg_computes_per_second=sum_computes_per_second/len(results)
    avg_computes_per_second_per_square_meter=sum_computes_per_second_per_square_meter/len(results)
    avg_tops=tops/len(results) 

    #print('total_energy in J:',total_energy)
    #print('total_area in cm2:',total_area)
    #print('total_latency in s:',total_latency)
    #print('total_mac:',total_mac)
    #print('avg_sum_tops_per_mm2:',avg_sum_tops_per_mm2)
    #print('avg_sum_tops_per_w:',avg_sum_tops_per_w)
    #print('avg_computes_per_second',avg_computes_per_second)
    #print('avg_computes_per_second_per_square_meter',avg_computes_per_second_per_square_meter)
    #print('max_percent_utilization:',max_percent_utilization)
    #print('area_utilized in cm2:',max_percent_utilization*total_area)
    return total_energy, total_area, total_latency, max_percent_utilization, avg_sum_tops_per_mm2, avg_sum_tops_per_w, max_tops_per_w, max_tops_per_mm2, avg_tops, accuracy




def checkIfallDictExist(dnn_names,file_store_paths, datakey):
    for dnn_name in dnn_names:
        dict_load=load_dict_from_json(file_store_paths[dnn_name])
        if search_instance_in_dict(dict_load, datakey)==False:
            a=False
            break
        if dnn_name == dnn_names[-1]:
            a=True
    if a==True:
        return True
    else:
        return False
# Directory you want to create

def run_several_iterations_separate(dnn_name, num_generations, population_size, num_iterations_to_run, seed_alg1, areaConstrcm2,typeObj,directory, num_mapping,n_jobs,prob_cross,eta_cross,prob_mut,eta_mut):
    if seed_alg1==None:
        seed_alg=None
    else:
        seed_alg=seed_alg1
    for testnumber in range(num_iterations_to_run):
        print('Iteration', testnumber, 'out of', num_iterations_to_run)
        #testnumber=0


        fileMeanStat=directory+'MeanStat_'+'test'+str(testnumber)+'.json' 
        resultsfile='test'+str(testnumber)+'.json'
        file_store_path=directory+'/'+resultsfile
        resultsfile2='Constrtest'+str(testnumber)+'.json'
        file_store_pathConstr=directory+'/'+resultsfile2
        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        coeff_lat=1.
        coeff_en=1.
        coeff_ar=1.
        coeff_acc=1.
        
        def objective(latency, energy, area, accuracy, coeff_lat, coeff_en, coeff_ar, coeff_acc, typeObj):
            if typeObj=='e':
                obj=energy
            elif typeObj=='l':
                obj=latency
            elif typeObj=='a':
                obj=area
            elif typeObj=='ela':
                obj=latency*energy*area
            elif typeObj=='el': 
                obj=latency*energy
            elif typeObj=='e_acc':
                obj=energy/accuracy
            elif typeObj=='l_acc':
                obj=latency/accuracy
            elif typeObj=='a_acc':
                obj=area/accuracy
            elif typeObj=='ela_acc':
                obj=latency*energy*area/accuracy
            elif typeObj=='el_acc': 
                obj=latency*energy/accuracy
            return obj

        empty_dict = {}
        # Store the empty dictionary to a JSON file
        with open(file_store_path, 'w') as f:
            json.dump(empty_dict, f, indent=4)
        with open(file_store_pathConstr, 'w') as f:
            json.dump(empty_dict, f, indent=4)
        with open(fileMeanStat, 'w') as file:
            json.dump([], file)
            
        voltages = [x / 100 for x in range(50, 90, 5)] #8
        bits_per_cell=[1,2,3,4,5,6,7,8]
        base_latency = [1e-9,2e-9,4e-9,6e-9,8e-9,1e-8]
        crossbar_sizeX=[32, 64, 128, 256, 512] #for the main crossbar
        
        crossbar_sizeY=[32, 64, 128, 256]
        shared_router_groupSize=[1,2,4,8,16,32]
       
        tiles_in_chip=[4,8,16,32,64]
        
        macros_in_tile=[4,8,16,32,64,128]
        
        glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [0.5,1,2,4,8,16,32]]

        # SOFTWARE VALUES
        # expansion factor:
        e0=[4.0, 4.5, 5.0, 5.5, 6.0]
        # 7 values
        e1_4=[4.0, 4.333333333333333, 4.666666666666667, 5.0, 5.333333333333333, 5.666666666666667, 6.0]
        # 11 values
        e5_8=[4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]
        # 21 val
        e9_12=[4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
        # 25 val
        e13_16=[4.0, 4.083333333333333, 4.166666666666667, 4.25, 4.333333333333333, 4.416666666666667, 4.5, 4.583333333333333, 4.666666666666667, 4.75, 4.833333333333333, 4.916666666666667, 5.0, 5.083333333333333, 5.166666666666667, 5.25, 5.333333333333333, 5.416666666666667, 5.5, 5.583333333333333, 5.666666666666667, 5.75, 5.833333333333333, 5.916666666666667, 6.0]
        # 49 val
        e17_20=[4.0, 4.041666666666667, 4.083333333333333, 4.125, 4.166666666666667, 4.208333333333333, 4.25, 4.291666666666667, 4.333333333333333, 4.375, 4.416666666666667, 4.458333333333333, 4.5, 4.541666666666667, 4.583333333333333, 4.625, 4.666666666666667, 4.708333333333333, 4.75, 4.791666666666667, 4.833333333333333, 4.875, 4.916666666666667, 4.958333333333333, 5.0, 5.041666666666667, 5.083333333333333, 5.125, 5.166666666666667, 5.208333333333333, 5.25, 5.291666666666667, 5.333333333333333, 5.375, 5.416666666666667, 5.458333333333333, 5.5, 5.541666666666667, 5.583333333333333, 5.625, 5.666666666666667, 5.708333333333333, 5.75, 5.791666666666667, 5.833333333333333, 5.875, 5.916666666666667, 5.958333333333333, 6.0]

        # kernel sizes:
        ks=[3, 5, 7]
        # 4 x 21 of these
        qs=[4, 6, 8]
        # we have 6 ds and 21 of other things
        d=[2,3,4]
        
        
        import numpy as np
        from pymoo.core.sampling import Sampling
        
        
        def getCBsize(lst,dnn_name,layer_name, w_bits):
            designspecs=getSpecs(dnn_name,layer_name)
            ins=designspecs.problem.instance
            C,R,S,M=ins["C"],ins["R"],ins["S"],ins["M"]
            rows=C*R*S
            cols=M*math.ceil(w_bits/bits_per_cell[lst[1]])
            total_cb=rows*cols
            return total_cb

        
        def checkSizeParallel(lst,dnn_name,layers,data_bits):
            total_cb=0
            
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(getCBsize)(lst,dnn_name,layer_name, data_bits[layer_name][0])
                for layer_name in layers
            )
            total_cb=sum(results)
            actual_cbs=crossbar_sizeX[lst[3]]*crossbar_sizeY[lst[4]]*shared_router_groupSize[lst[5]]*tiles_in_chip[lst[6]]*macros_in_tile[lst[7]]
            if actual_cbs>=total_cb:
                return True
            else:
                return False
                
        
        
            
        def total_requiredParallel(lst,dnn_name,layers, data_bits):
            total_cb=0
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(getCBsize)(lst,dnn_name,layer_name, data_bits[layer_name][0])
                for layer_name in layers
            )
            total_cb=sum(results)
            return total_cb  
            
        def actual_present(lst,dnn_name):
            actual_cbs=crossbar_sizeX[lst[3]]*crossbar_sizeY[lst[4]]*shared_router_groupSize[lst[5]]*tiles_in_chip[lst[6]]*macros_in_tile[lst[7]]
            return actual_cbs 
        

        def to_APQ(x):
            d_samlist=[]
            for ii2 in range(9, 15):
                d_samlist.append(d[x[ii2]])
            ks_samlist=[]
            for ii2 in range(15, 36):
                ks_samlist.append(ks[x[ii2]])
            q_pw_w_samlist=[]
            for ii2 in range(36, 57):
                q_pw_w_samlist.append(qs[x[ii2]])
            q_pw_a_samlist=[]
            for ii2 in range(57, 78):
                q_pw_a_samlist.append(qs[x[ii2]])
            q_dw_w_samlist=[]
            for ii2 in range(78, 99):
                q_dw_w_samlist.append(qs[x[ii2]])
            q_dw_a_samlist=[]
            for ii2 in range(99, 120):
                q_dw_a_samlist.append(qs[x[ii2]])
            e_samlist=[]    
            e_samlist.append(e0[x[120]])
            for ii2 in range(121, 125):
                e_samlist.append(e1_4[x[ii2]])
            for ii2 in range(125, 129):
                e_samlist.append(e5_8[x[ii2]])
            for ii2 in range(129, 133):
                e_samlist.append(e9_12[x[ii2]])
            for ii2 in range(133, 137):
                e_samlist.append(e13_16[x[ii2]])
            for ii2 in range(137, 141):
                e_samlist.append(e17_20[x[ii2]])
            #print(len(e_samlist))
            return d_samlist,ks_samlist,e_samlist,q_pw_w_samlist,q_pw_a_samlist,q_dw_w_samlist,q_dw_a_samlist
        
        class MySampling(FloatRandomSampling):
            def _do(self, problem, n_samples, **kwargs):
                n, (xl, xu) = problem.n_var, problem.bounds()
                final_output=[]
                
                startsamp=time.time()
                while len(final_output)<n_samples:
                    fullfilled=len(final_output)
                    output= np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples-fullfilled) for k in range(n)])
                    for i in range(output.shape[0]):
                        #
                        d_samlist,ks_samlist,e_samlist,q_pw_w_samlist,q_pw_a_samlist,q_dw_w_samlist,q_dw_a_samlist = to_APQ(output[i])

                        CIMNAS_software(d_samlist,ks_samlist,e_samlist,q_pw_w_samlist,q_pw_a_samlist,q_dw_w_samlist,q_dw_a_samlist)

                        layers = [f for f in os.listdir(f"../main/workloads/{dnn_name}") if f != "index.yaml" and f.endswith(".yaml")]
                        layers = sorted(l.split(".")[0] for l in layers)
                        quant_file='workloads/mobilenet_v2/bits.json'
                        with open(quant_file, 'r') as file:
                            data_bits = json.load(file)
                        check=checkSizeParallel(output[i],dnn_name,layers, data_bits)
                        if check==True:
                            final_output.append(output[i])
               
                finishsamp=time.time()
                print('Sampling done, took',finishsamp-startsamp,' sec')
                return np.row_stack([final_output[kk] for kk in range(len(final_output))])


        
        class MyProblem(Problem):
        
            def __init__(self):
                
                super().__init__(n_var=141, n_obj=1, n_ieq_constr=2, 
                                 xl=[0,0,0,0,0,0,0,0,0, # for hardware (x[0]-x[8])
                                     0,0,0,0,0,0, #for d (x[9] - x[14])
                                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, #for ks (x[15] - x[35])
                                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, #for pw_w (x[36] - x[56])
                                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, #for pw_a (x[57] - x[77])
                                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, #for dw_w (x[78] - x[98])
                                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, #for dw_a (x[99] - x[119])
                                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 #for e (x[120] - x[140])
                                    ], 
                                 xu=[7,7,5,4,3,5,4,5,6, # for hardware
                                     #7,7,5,4,4,5,5,6,6, # for hardware old
                                     2,2,2,2,2,2, #for d
                                     2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, #for ks
                                     2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, #for pw_w
                                     2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                     2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                     2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, #for dw_a
                                     4,6,6,6,6,10,10,10,10,20,20,20,20,24,24,24,24,48,48,48,48
                                    ], 
                                 vtype=int)
                # hardware - d - ks - pw_w - pw_a - dw_w - dw_a - e 
                # the first 9 variables are hardware variables
                # new variables:
                # 6 for "d" (3 val)
                # 21 for "ks" (3 val)
                # 21 for "e" --> this is tricky
                # order here: q_pw_w_sam, q_pw_a_sam, q_dw_w_sam, q_dw_a_sam
                # 21 for pw_a (3 val)
                # 21 for dw_a (3 val)
                # 21 for pw_w (3 val)
                # 32 for dw_w (3 val)
                # for e: 
                    # layer 0 --> 5 val
                    # lr 1-4 --> 7 val
                    # lr 5-8 --> 11 val
                    # lr 9-12 --> 21 val
                    # lr 13-16 --> 25 val
                    # lr 17-20 --> 49 val
                
            def _evaluate(self, x, out, *args, **kwargs):
                #print('Before')
                #print(x)
                #newxxx=eliminateWrongSizes(x)
                #print('After')
               # print(newxxx)
                #print('**********x')
                #print(x)
                startt=time.time()
                new_x=np.zeros(x.shape)
                new_f=np.zeros(x.shape[0])
                constrs=2
                new_g=np.zeros((x.shape[0],constrs))
                #print(range(x.shape[0]))
                d_sam, ks_sam, e_sam, q_pw_w_sam, q_pw_a_sam, q_dw_w_sam, q_dw_a_sam = {},{},{},{},{},{},{}
                d_dict, ks_dict, qdict1, qdict2, qdict3, qdict4, e_dict = {},{},{},{},{},{},{}
                for i in range(x.shape[0]):
                    #print(x[i])
                    #print(C[x[i][0]])
                    #print(Y[x[i][1]])
                    new_x[i][0]=voltages[x[i][0]]
                    new_x[i][1]=bits_per_cell[x[i][1]]
                    new_x[i][2]=base_latency[x[i][2]]
                    new_x[i][3]=crossbar_sizeX[x[i][3]]
                    new_x[i][4]=crossbar_sizeY[x[i][4]]
                    new_x[i][5]=shared_router_groupSize[x[i][5]]
                    new_x[i][6]=tiles_in_chip[x[i][6]]
                    new_x[i][7]=macros_in_tile[x[i][7]]
                    new_x[i][8]=glb_buffer_depth[x[i][8]]
                    
                    d_samlist=[]
                    d_dict_this=''
                    for ii2 in range(9, 15):
                        new_x[i][ii2]=int(d[x[i][ii2]])
                        d_samlist.append(new_x[i][ii2])
                        d_dict_this=d_dict_this+str(int(new_x[i][ii2]))
                    d_sam[str(i)]=d_samlist
                    d_dict[str(i)]=d_dict_this
                    
                    ks_samlist=[]
                    ks_dict_this=''
                    for ii2 in range(15, 36):
                        new_x[i][ii2]=int(ks[x[i][ii2]])
                        ks_samlist.append(new_x[i][ii2])
                        ks_dict_this=ks_dict_this+str(int(new_x[i][ii2]))
                    ks_sam[str(i)]=ks_samlist
                    ks_dict[str(i)]=ks_dict_this
                    
                    q_pw_w_samlist=[]
                    qdict1_this=''
                    for ii2 in range(36, 57):
                        new_x[i][ii2]=int(qs[x[i][ii2]])
                        q_pw_w_samlist.append(new_x[i][ii2])
                        qdict1_this=qdict1_this+str(int(new_x[i][ii2]))
                    q_pw_w_sam[str(i)]=q_pw_w_samlist
                    qdict1[str(i)]=qdict1_this
                    
                    q_pw_a_samlist=[]
                    qdict2_this=''
                    for ii2 in range(57, 78):
                        new_x[i][ii2]=int(qs[x[i][ii2]])
                        q_pw_a_samlist.append(new_x[i][ii2])
                        qdict2_this=qdict2_this+str(int(new_x[i][ii2]))
                    q_pw_a_sam[str(i)]=q_pw_a_samlist
                    qdict2[str(i)]=qdict2_this
                    
                    q_dw_w_samlist=[]
                    qdict3_this=''
                    for ii2 in range(78, 99):
                        new_x[i][ii2]=int(qs[x[i][ii2]])
                        q_dw_w_samlist.append(new_x[i][ii2])
                        qdict3_this=qdict3_this+str(int(new_x[i][ii2]))
                    q_dw_w_sam[str(i)]=q_dw_w_samlist
                    qdict3[str(i)]=qdict3_this
                    
                    q_dw_a_samlist=[]
                    qdict4_this=''
                    for ii2 in range(99, 120):
                        new_x[i][ii2]=int(qs[x[i][ii2]])
                        q_dw_a_samlist.append(new_x[i][ii2])
                        qdict4_this=qdict4_this+str(int(new_x[i][ii2]))
                    q_dw_a_sam[str(i)]=q_dw_a_samlist
                    qdict4[str(i)]=qdict4_this

                    e_dictthis=''
                    e_samlist=[]    
                    new_x[i][120]=e0[x[i][120]]
                    e_samlist.append(new_x[i][120])
                    #e_dictthis=e_dictthis+str(round(new_x[i][120], 2))
                    e_dictthis=e_dictthis+str(int(x[i][120]))
                    e_dictthis=e_dictthis+'_'
                    for ii2 in range(121, 125):
                        new_x[i][ii2]=e1_4[x[i][ii2]]
                        e_samlist.append(new_x[i][ii2])
                        #e_dictthis=e_dictthis+str(round(new_x[i][ii2], 2))
                        e_dictthis=e_dictthis+str(int(x[i][ii2]))
                        e_dictthis=e_dictthis+'_'
                    for ii2 in range(125, 129):
                        new_x[i][ii2]=e5_8[x[i][ii2]]
                        e_samlist.append(new_x[i][ii2])
                        #e_dictthis=e_dictthis+str(round(new_x[i][ii2], 2))
                        e_dictthis=e_dictthis+str(int(x[i][ii2]))
                        e_dictthis=e_dictthis+'_'
                    for ii2 in range(129, 133):
                        new_x[i][ii2]=e9_12[x[i][ii2]]
                        e_samlist.append(new_x[i][ii2])
                        #e_dictthis=e_dictthis+str(round(new_x[i][ii2], 2))
                        e_dictthis=e_dictthis+str(int(x[i][ii2]))
                        e_dictthis=e_dictthis+'_'
                    for ii2 in range(133, 137):
                        new_x[i][ii2]=e13_16[x[i][ii2]]
                        e_samlist.append(new_x[i][ii2])
                        #e_dictthis=e_dictthis+str(round(new_x[i][ii2], 2))
                        e_dictthis=e_dictthis+str(int(x[i][ii2]))
                        e_dictthis=e_dictthis+'_'
                    for ii2 in range(137, 141):
                        new_x[i][ii2]=e17_20[x[i][ii2]]
                        e_samlist.append(new_x[i][ii2])
                        #e_dictthis=e_dictthis+str(round(new_x[i][ii2], 2))
                        e_dictthis=e_dictthis+str(int(x[i][ii2]))
                        if ii2!=140:
                            e_dictthis=e_dictthis+'_'
                    e_sam[str(i)]=e_samlist
                    e_dict[str(i)]=e_dictthis

                for i in range(new_x.shape[0]):
                    hardware_params=[new_x[i][0],new_x[i][1],new_x[i][2],new_x[i][3],new_x[i][4],new_x[i][5],new_x[i][6],new_x[i][7],new_x[i][8]]

                    # here we need to run whole python first for each i 
                    #(superneetwork part and accuracy estimation)
                    #print(d_sam[str(i)])
                    #print(type(d_sam[str(i)]))
                    CIMNAS_software([int(num) for num in d_sam[str(i)]], [int(num) for num in ks_sam[str(i)]], e_sam[str(i)],[int(num) for num in q_pw_w_sam[str(i)]],[int(num) for num in q_pw_a_sam[str(i)]] ,[int(num) for num in q_dw_w_sam[str(i)]],[int(num) for num in q_dw_a_sam[str(i)]] )
                    #layers = [f for f in os.listdir(f"../models/workloads/{DNN}") if f != "index.yaml" and f.endswith(".yaml")]
                    layers = [f for f in os.listdir(f"../main/workloads/{dnn_name}") if f != "index.yaml" and f.endswith(".yaml")]
                    layers = sorted(l.split(".")[0] for l in layers)
                    quant_file='workloads/mobilenet_v2/bits.json'
                    with open(quant_file, 'r') as file:
                        data_bits = json.load(file)
                    
                    try:
                        
                        vl,bps,bl,csX,csY,srg,tic,mit,gbp=hardware_params
                        vl,bl =[float(x) for x in [vl,bl]]
                        write_to_dic=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
                        bps,csX,csY,srg,tic,mit,gbp=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
                        #for 
                        #write_to_dic=vl,bps,bl,csX,csY,srg,tic,mit,gbp
                        write_to_dic=vl,bps,bl,csX,csY,srg,tic,mit,gbp,d_dict[str(i)], ks_dict[str(i)], qdict1[str(i)], qdict2[str(i)], qdict3[str(i)], qdict4[str(i)], e_dict[str(i)]
                        datakey='A_'+'_'.join(map(str, write_to_dic))
                        #print('datakey',datakey)
                        dict_load=load_dict_from_json(file_store_path)
                        if search_instance_in_dict(dict_load, datakey)==True:
                            dataForScore=dict_load[datakey] # dictionary with data
                            total_latency=dataForScore['l']
                            total_energy=dataForScore['e']
                            total_area=dataForScore['a']
                            max_percent_utilization=dataForScore['u']
                            avg_sum_tops_per_mm2=dataForScore['a_tpm']
                            avg_sum_tops_per_w=dataForScore['a_tpw']
                            max_tops_per_w=dataForScore['m_tpw']
                            max_tops_per_mm2=dataForScore['m_tpm']
                            tops=dataForScore['tp']
                            accuracy=dataForScore['acc']
                        else:
                            
                            
                            check=checkSizeParallel(x[i],dnn_name,layers,data_bits)
                            if check==False:
                                total_energy, total_area, total_latency=1e25,1e25,1e25
                                accuracy=1e-25
                                
                            else:
                                
                                total_energy, total_area, total_latency, max_percent_utilization, avg_sum_tops_per_mm2, avg_sum_tops_per_w, max_tops_per_w, max_tops_per_mm2, tops, accuracy=getMetrics(hardware_params,dnn_name, num_mapping,n_jobs, layers, data_bits)
                                dictts={'e': total_energy, 'a': total_area, 'l':total_latency,'a_tpm':avg_sum_tops_per_mm2, 'a_tpw': avg_sum_tops_per_w,'u': max_percent_utilization,'au': max_percent_utilization*total_area, 'm_tpm':max_tops_per_mm2, 'm_tpw':max_tops_per_w,'tp':tops, 'acc': accuracy }
                                add_new_instance_if_not_exists(file_store_path, datakey, dictts)
                                if total_area<=areaConstrcm2:
                                   add_new_instance_if_not_exists(file_store_pathConstr, datakey, dictts) 
                                
                        areaforscore=total_area
                        score=objective(total_latency, total_energy, total_area, accuracy, coeff_lat, coeff_en, coeff_ar, coeff_acc, typeObj)    
                    
                    except RuntimeError:
                        
                        score=1e25 
                        areaforscore=1e25
                        
                        continue
                    new_f[i]=score
                    new_g[i][0]=total_requiredParallel(x[i],dnn_name,layers,data_bits)-actual_present(x[i],dnn_name)
                    new_g[i][1]=areaforscore-areaConstrcm2

                
                filtered_data = new_f[new_f < 1e15]
                # Calculate mean and standard deviation
                if len(filtered_data)==0:
                    mean_value=1e25
                    std_deviation =1e25
                    best = 1e25
                else:
                    mean_value = np.mean(filtered_data)
                    std_deviation = np.std(filtered_data)
                    best = min(filtered_data)

                
                fmeanAndStd=[mean_value,std_deviation,best]
                print(fmeanAndStd)
                with open(fileMeanStat, 'r') as file2:
                    datafm = json.load(file2)
                datafm.extend(list(list(fmeanAndStd)))
                with open(fileMeanStat, 'w') as file2:
                    json.dump(datafm, file2)    
                
                finisht=time.time()
                print('Finished 1 eval (1 gen), took',finisht-startt,' sec')
                out["G"] = new_g
                out["F"] = new_f
                
        
        print('Starting search')
        
        problem = MyProblem()
        start_time_search = time.time()
        method = GA(pop_size=population_size, #20
                    sampling=MySampling(),
                    crossover=SBX(prob=prob_cross, eta=eta_cross, vtype=float, repair=RoundingRepair()),
                    mutation=PM(prob=prob_mut, eta=eta_mut, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                    )
        
        res = minimize(problem,
                       method,
                       termination=('n_gen', num_generations), #20
                       seed=seed_alg, #was set to 1 to ensure repredusability
                       save_history=True,
                       verbose= True
                       )
        end_time_search = time.time()
        
        print('time in s:', end_time_search-start_time_search)