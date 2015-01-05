
"""
Autism Emotion Regulation Pipeline (Nipype 0.7)
- Level 1 in subject's own functional space
- Coregister output to freesurfer anatomy
- ANTS normalization done offline (ANTS_batch.sh, WIMT_batch.sh)
- Level 2 using ANTS normalized con images

Created:		03-5-2012	# based on Domain pipeline script L2 (J.A.R.)
Code Revised:	??-??-????
"""

import os                                    # system functions
import nipype.algorithms.modelgen as model   # model generation
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.io as nio           # i/o routines
import nipype.interfaces.matlab as mlab      # how to run matlab
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
from nipype.utils.filemanip import loadflat	 # some useful stuff for debugging
import scipy.io as sio
import numpy as np
from nipype.interfaces.base import Bunch
from copy import deepcopy
import sys

covariates=True
#from nipype.utils import config
#config.set('execution', 'remove_unnecessary_outputs', 'false')
#config.enable_debug_mode()

# Tell freesurfer what subjects directory to use
subjects_dir = '/Volumes/lashley/AutReg.01/Analysis/nipype/'
fs.FSCommand.set_default_subjects_dir(subjects_dir)

# Set the way matlab should be called
mlab.MatlabCommand.set_default_matlab_cmd("//Volumes/Macintosh_HD/Applications/MATLAB_R2011b.app//bin//matlab -nodesktop -nosplash")
#If SPM is not in your MATLAB path you should add it here
mlab.MatlabCommand.set_default_paths('/Volumes/lashley/packages/spm8/spm8/')
# Set up how FSL should write nifti files:
fsl.FSLCommand.set_default_output_type('NIFTI')



subjects_list_type = 'Long' #Short/Long
covariates_list = 'Long' #Short/Long

if subjects_list_type is 'Short':
	subjects_list=[		'20100922_11436',#C 70.66257778	68.19392897	71.57593689
				'20101029_11690',#C 27.59142857 24.06875	9.666666667
				'20101122_11835',#C 37.38285714 56.89		50.394
				'20101203_11919',#C 28.37 	45.57214286	28.66454545
				'20110114_12129',#C 73.41285714  53.97866667	78.48466667
				'20110119_12153',#C 85.44142857	77.750625	83.812
				'20110216_12271',#C 77.94857143	77.280625	77.432
				'20110317_12453',#C 85.44142857	77.750625	83.812
				'20110325_12521',#C 32.26857143	47.83769231	41.86769231
				'20110329_12555',#C 84.65857143	95.57375	92.64866667
				'20110412_12669',#C 37.54	43.19928571	39.10307692
				'20110426_12769',#C 94.75142857	91.525625	94.13
				'20110502_12821',#C 93.79571429	86.13125	80.14866667
				'20110519_12925',#C 70.66257778	68.19392897	71.57593689
				'20101028_11683',#A 73.668  	68.935		67.28
				'20101101_11706',#A 97.334 	96.84916667	96.85272727
				'20101111_11783',#A 89.19666667 69.305		69.53071429
				'20101117_11814',#A 84.84285714 82.195		81.90333333
				'20101117_11816',#A 87.108 	88.49166667	90.85454545
				'20101130_11879',#A 0 		3.153333333	48.61
				'20110107_12092',#A 63.67571429 61.106		56.55714286
				'20110114_12131',#A 21 		32.29583333	26.56090909
				'20110118_12140',#A 76.74 	79.99875	75.68
				'20110304_12363',#A 73.668  	68.935		67.28
				'20110311_12414',#A 95.21857143 95.135		95.14266667
				'20110404_12600',#A  89.99857143 87.03		66.62357143
				'20110429_12799'#A 65.51 	72.36266667	81.10923077
				
				]
				
																					
				

				
	cons=		[	'20100922_11436',#C
				'20101029_11690',#C
				'20101122_11835',#C
				'20101203_11919',#C
				'20110114_12129',#C
				'20110119_12153',#C
				'20110216_12271',#C
				'20110317_12453',#C
				'20110325_12521',#C
				'20110329_12555',#C
				'20110412_12669',#C
				'20110426_12769',#C
				'20110502_12821',#C
				'20110519_12925',#C
			]
	auts=		[	'20101028_11683',#A
				'20101101_11706',#A
				'20101111_11783',#A
				'20101117_11814',#A
				'20101117_11816',#A
				'20101130_11879',#A
				'20110107_12092',#A
				'20110114_12131',#A
				'20110118_12140',#A
				'20110304_12363',#A
				'20110311_12414',#A
				'20110404_12600',#A
				'20110429_12799'#A
				
				]
if subjects_list_type is 'Long':
	subjects_list=		[
			
				'20101011_11549',#C 70.66257778	68.19392897	71.57593689
				'20100922_11436',#C 70.66257778	68.19392897	71.57593689
				'20101029_11690',#C 27.59142857 24.06875	9.666666667
				'20101122_11835',#C 37.38285714 56.89		50.394
				'20101203_11919',#C 28.37 	45.57214286	28.66454545
				'20110114_12129',#C 73.41285714  53.97866667	78.48466667
				'20110119_12153',#C 85.44142857	77.750625	83.812
				'20110216_12271',#C 77.94857143	77.280625	77.432
				'20110317_12453',#C 95.79285714	96.62733333	95.64076923
				'20110325_12521',#C 32.26857143	47.83769231	41.86769231
				'20110329_12555',#C 84.65857143	95.57375	92.64866667
				'20110412_12669',#C 37.54	43.19928571	39.10307692
				'20110426_12769',#C 94.75142857	91.525625	94.13
				'20110502_12821',#C 93.79571429	86.13125	80.14866667
				'20110519_12925',#C 70.66257778	68.19392897	71.57593689
				'20110310_12406',#C 70.66257778	68.19392897	71.57593689
				'20101028_11683',#A 77.20428571 83.304375	89.18466667
				'20101101_11706',#A 97.334 	96.84916667	96.85272727
				'20101111_11783',#A 89.19666667 69.305		69.53071429
				'20101117_11814',#A 84.84285714 82.195		81.90333333
				'20101117_11816',#A 87.108 	88.49166667	90.85454545
				'20101130_11879',#A 0 		3.153333333	48.61
				'20110107_12092',#A 63.67571429 61.106		56.55714286
				'20110114_12131',#A 21 		32.29583333	26.56090909
				'20110118_12140',#A 76.74 	79.99875	75.68
				'20110304_12363',#A 73.668  	68.935		67.28
				'20110311_12414',#A 95.21857143 95.135		95.14266667
				'20110404_12600',#A 89.99857143 87.03		66.62357143
				'20110429_12799',#A 65.51 	72.36266667	81.10923077
				'20110224_12334',#A 77.73  	43.86714286	52.665
				'20110315_12437'#A 60.712  	58.88		75.08454545
		 
				 	 																		
				]
	
	
	cons =			[
	
				'20101011_11549',#C1
				'20100922_11436',#C2
				'20101029_11690',#C3
				'20101122_11835',#C4
				'20101203_11919',#C5
				'20110114_12129',#C6
				'20110119_12153',#C7
				'20110216_12271',#C8
				'20110317_12453',#C9
				'20110325_12521',#C10
				'20110329_12555',#C11
				'20110412_12669',#C12
				'20110426_12769',#C13
				'20110502_12821',#C14
				'20110519_12925',#C15
				'20110310_12406',#C16
				
				]
	
	auts= 			[
	
				'20101028_11683',#A1
				'20101101_11706',#A2
				'20101111_11783',#A3
				'20101117_11814',#A4
				'20101117_11816',#A5
				'20101130_11879',#A6
				'20110107_12092',#A7
				'20110114_12131',#A8
				'20110118_12140',#A9
				'20110304_12363',#A10
				'20110311_12414',#A11
				'20110404_12600',#A12
				'20110429_12799',#A13
				'20110224_12334',#A14
				'20110315_12437'#A15
				
				]
	


#indicate group1 and group 2, change when appropriate
myGroup1 = cons  
myGroup2 = auts


"""
Level 2 Pipeline -- ANTS normalized anatomy and con images
"""
def ordersubjects(files, subj_list):
    import sys
    outlist = []
    for s in subj_list:
        subj_found = False
        for f in files:
	    #print f
            if '%s'%s in f:
                outlist.append(f)
                subj_found = True
                continue
        if subj_found == False:
            # Fail hard if expected con images are missing
            sys.stderr.write("Con images for subject %s could not be found!"%(s))
            sys.exit("Con images for subject %s could not be found!"%(s))
    print '===============',outlist
    return outlist

def list2tuple(listoflist):
    return [tuple(x) for x in listoflist]



if covariates is True:
	l2pipeline=pe.Workflow(name='l2output_independent-samples_covariates_FTEST')
if covariates is False:
	l2pipeline = pe.Workflow(name='l2output_independent-samples_FTEST')


# Input node for second level (group analysis) pipeline
l2inputnode = pe.Node(

	interface					=util.IdentityInterface(fields=['contrasts']),
	
				iterables 		= [('contrasts', range(4,4+1))],
				name			='inputnode')


#datagrabber
# Source information for group analysis data
l2source = pe.Node(

		interface=nio.DataGrabber(

			infields			=['l1con_id'],
			outfields			=['l1con']),
			
					name		='l2source')
					
					
l2source.inputs.base_directory = os.path.abspath('/Volumes/lashley/AutReg.01/Analysis/nipype/l1pipeline/')
l2source.inputs.template = '*'
l2source.inputs.field_template = dict(l1con='/Volumes/lashley/AutReg.01/Analysis/nipype/l1pipeline/_subject_id_*/warp_con/mapflow/_warp_con*/ess_%04d_wimt.img')
l2source.inputs.template_args = dict(l1con=[['l1con_id']])



# setup a 2-sample t-test
twosamplettestdes = pe.Node(interface=spm.TwoSampleTTestDesign(), name="twosamplettestdes")
twosamplettestdes.inputs.explicit_mask_file = os.path.abspath('/Volumes/lashley/AutReg.01/Analysis/ANTS/MNI152_T1_1mm_brain_uncompressed.nii')

if covariates is True:
	if covariates_list is 'Long':
		
		twosamplettestdes.inputs.covariates = [	
		
		
		dict(vector=[	70.66257778,70.66257778	,27.59142857,37.38285714, 28.37,73.41285714,85.44142857,77.94857143,95.79285714,32.26857143,84.65857143,37.54,	94.75142857,	93.79571429,70.66257778,70.66257778,77.20428571,	97.334 ,	89.19666667,	84.84285714 ,87.108 ,0 ,		63.67571429,	21 ,	76.74 	,73.668 , 95.21857143	,	89.99857143 	,65.51 ,77.73 , 60.712  		],	name='look'),
	
		dict(vector=[	68.19392897,68.19392897, 24.06875,56.89,45.57214286,53.97866667,77.750625,77.280625,96.62733333,47.83769231,95.57375,	43.19928571,	91.525625,	86.13125,68.19392897,68.19392897,83.304375,	96.84916667,	69.305	,	82.195	,	88.49166667,	3.153333333, 	61.106	,32.29583333,79.99875,	68.935	, 95.135,87.03,72.36266667,43.86714286,58.88			],	name='faceneg'),
	
		dict(vector=[	71.57593689,71.57593689,9.666666667,50.394,28.66454545,	78.48466667,83.812,77.432,95.64076923,41.86769231,92.64866667,	39.10307692,	94.13,	80.14866667,71.57593689,71.57593689	,89.18466667,	96.85272727,	69.53071429,	81.90333333,	90.85454545,	48.61,		56.55714286,26.56090909,75.68	,	67.28,95.14266667,66.62357143,81.10923077,52.665,	75.08454545			],	name='facepos'),
							
		]
		
		

		
	if covariates_list is 'Short':
		
		twosamplettestdes.inputs.covariates = [	
		
		dict(vector=[	70.66257778,
				27.59142857, 
				37.38285714, 
				28.37 ,
				73.41285714, 
				85.44142857,
				77.94857143,
				85.44142857,
				32.26857143,
				84.65857143,
				37.54,
				94.75142857,
				93.79571429,
				70.66257778, #C
				73.668 ,
				97.334, 
				89.19666667 ,
				84.84285714 ,
				87.108, 
				0,
				63.67571429,
				21 ,
				76.74 ,
				73.668,
				95.21857143,
				89.99857143,
				65.51 			],	name='look'),
		
		dict(vector=[	68.19392897, 	#C1
				24.06875, 	#C2
				56.89,		#C3
				45.57214286,	#C4
				53.97866667,	#C5
				77.750625, 	#C6
				77.280625,	#C7
				77.750625,	#C8
				47.83769231,	#C9
				95.57375,	#C10
				43.19928571,	#C11
				91.525625,	#C12
				86.13125,	#C13
				68.19392897,	#C14
				68.935,		#A1
				96.84916667,	#A2
				69.305,		#A3
				82.195,		#A4
				88.49166667,	#A5
				3.153333333,	#A6
				61.106,		#A7
				32.29583333,	#A8				
				79.99875,	#A9
				68.935,		#A10
				95.135,		#A11
				87.03,		#A12
				72.36266667	#A13	
				
						],	name='faceneg'),
		
		dict(vector=[	71.57593689, 	#C1
				9.666666667,	#C2
				50.394	,	#C3
				28.66454545,	#C4
				78.48466667, 	#C5
				83.812, 	#C6
				77.432,	  	#C7
				83.812,  	#C8
				41.86769231, 	#C9
				92.64866667, 	#C10
				39.10307692, 	#C11
				94.13, 		#C12
				80.14866667,	#C13
				71.57593689,	#C14
				67.28,		#A1
				96.85272727,	#A2
				69.53071429,	#A3
				81.90333333 ,	#A4
				90.85454545,	#A5
				48.61,		#A6
				56.55714286 ,	#A7
				26.56090909,	#A8
				75.68,		#A9
				67.28,		#A10
				95.14266667,	#A11
				66.62357143,	#A12
				81.10923077	#A13
						
						],	name='facepos'),
		
		]


			

l2estimate = pe.Node(interface=spm.EstimateModel(), name="level2estimate")
l2estimate.inputs.estimation_method = {'Classical' : 1}





l2conestimate = pe.Node(interface = spm.EstimateContrast(), name="level2conestimate")
L2cont1 = ('Group1 Mean','T', ['Group_{1}','Group_{2}'],[1,0])
L2cont2 = ('Group1 -Mean','T', ['Group_{1}','Group_{2}'],[-1,0])
L2cont3 = ('Group2 Mean','T', ['Group_{1}','Group_{2}'],[0,1])
L2cont4 = ('Group2 -Mean','T', ['Group_{1}','Group_{2}'],[0,-1])
L2cont5 = ('Group1 > Group2','T', ['Group_{1}','Group_{2}'],[1,-1])
L2cont6 = ('Group2 > Group1','T', ['Group_{1}','Group_{2}'],[-1,1])
L2cont7 = ('Group1+Group2 Mean','T', ['Group_{1}','Group_{2}'],[0.5,0.5])
l2conestimate.inputs.contrasts = [L2cont1, L2cont2, L2cont3, L2cont4, L2cont5, L2cont6, L2cont7]
l2conestimate.inputs.group_contrast = True


l2FDRthresh = pe.MapNode(interface = spm.Threshold(), name="level2FDRthreshold", iterfield = ['stat_image','contrast_index'])
l2FDRthresh.iterables = [('height_threshold', [0.05, 0.01, 0.001, 0.0001])]
l2FDRthresh.inputs.extent_fdr_p_threshold = 0.05
l2FDRthresh.inputs.extent_threshold = 0
l2FDRthresh.inputs.contrast_index = [1,2,3,4,5,6,7]
l2FDRthresh.inputs.use_fwe_correction = False
l2FDRthresh.inputs.use_topo_fdr = True




l2pipeline.base_dir = os.path.abspath('/Volumes/lashley/AutReg.01/Analysis/nipype/')
l2pipeline.connect([
					(l2inputnode,l2source,[('contrasts','l1con_id')]),
					(l2source,twosamplettestdes,[(('l1con',ordersubjects,myGroup1),'group1_files'),
					
											  (('l1con',ordersubjects,myGroup2),'group2_files')]),
					(twosamplettestdes,l2estimate,[('spm_mat_file','spm_mat_file')]),
					(l2estimate,l2conestimate,[('spm_mat_file','spm_mat_file'),
												('beta_images','beta_images'),
												('residual_image','residual_image')]),
					(l2conestimate,l2FDRthresh,[('spm_mat_file','spm_mat_file'),
												('spmT_images','stat_image')]),
                    ])
l2pipeline.config['execution'] = {'remove_unnecessary_outputs':'False'}





#l2pipeline.write_graph()
l2pipeline.run(plugin='MultiProc', plugin_args={'n_procs' : 12})
