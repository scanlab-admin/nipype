"""
Import modules
"""

import os                                    # system functions
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.io as nio           # i/o routines
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl          # fsl module


"""
Define experiment specific parameters
"""

#to better access the parent folder of the experiment
experiment_dir = '/Volumes/lashley/DotPro.01/Analysis/nipype/'

#dirnames for functional ROI and of level1 datasink
fROIOutput = 'fROI_output'      #name of fROI datasink
l1contrastDir = '2nd_level_VOL_results' #name of first level datasink

#list of subjectnames
subjects = ['SAD_C19','SAD_C20','SAD_C21','SAD_C22','SAD_C23','SAD_C24','SAD_C25','SAD_C26','SAD_C27','SAD_C28','SAD_C29','SAD_C30','SAD_C31','SAD_C32','SAD_C33','SAD_C34','SAD_C35','SAD_C36','SAD_C37','SAD_C38','SAD_C39','SAD_C40','SAD_C41','SAD_C42','SAD_P04','SAD_P05','SAD_P06','SAD_P07','SAD_P08','SAD_P09','SAD_P11','SAD_P12','SAD_P13','SAD_P14','SAD_P15','SAD_P16','SAD_P17','SAD_P19','SAD_P20','SAD_P21','SAD_P22','SAD_P23','SAD_P24','SAD_P25','SAD_P26','SAD_P27','SAD_P28','SAD_P29','SAD_P30','SAD_P31']

def getSegVersion(in_file, version):
    if version == 0:
       return in_file[0]
    else:
       return in_file[1]
    
def ordersubjects(files, subj_list):
	outlist = []
	for subj in subj_list:
		for file in files:	
			if '%s' %subj in file:
				outlist.append(file)
				continue
	return outlist

def pathfinder(subject):
	import os
	anat_dir = '/Volumes/lashley/DotPro.01/Data/anat/%s/anat.nii.gz' %(subject)
	return anat_dir

# Tell freesurfer what subjects directory to use
subjects_dir 			= experiment_dir
fs.FSCommand.set_default_subjects_dir(subjects_dir)

#dirnames for anatomical ROI pipeline
aROIOutput 				= '_anat_ROI_output'      #name of aROI datasink
l1contrastDir 				= '2nd_level_VOL_results' #name of second level datasink


#list of contrastnumbers the pipeline should consider
contrasts 				= ['01','02','04','05','07','08','10','11','13','14','16','17'] #these are all the event vs fix contrasts

#name of the first session from the first level pipeline
nameOfFirstSession 			= 'f'

ROIregionsorig 				= ['18','54','2031','1031']
ROIregions2009 				= []



#Initiation of the ROI extraction workflow
aROIflow = pe.Workflow(

	name				='_anat_ROIs')
	
aROIflow.base_dir = experiment_dir

inputnode = pe.Node(

	interface			=util.IdentityInterface(
	
		fields			=['subject_id','contrast_id']),

			 name		='inputnode')
			 
inputnode.iterables 			= [('subject_id', subjects),
                    			   ('contrast_id', contrasts)]


#Node: DataGrabber - to grab the input data
datagrabber = pe.Node(

	interface			=nio.DataGrabber(
	
		infields		=['subject_id','contrast_id'],
		outfields		=['contrast','bb_id']),
		
			name 		= 'datagrabber')
			
			
datagrabber.inputs.base_directory 	= experiment_dir + '/2nd_level_VOL_results/'
datagrabber.inputs.template 		= experiment_dir + '/2nd_level_VOL_results/'+'/_subject_id_%s/%s/%s%s%s'

info 					= dict(contrast = [['subject_id','_mri_convert2nii*','con_00','contrast_id','_out.nii']],
           				       bb_id = [['subject_id','_bbregister0','rf_dtype_out_st_bbreg_','subject_id','.dat']])
				 
datagrabber.inputs.template_args 	= info





#Node: FreeSurferSource - to grab FreeSurfer files from the recon-all process
fssource = pe.Node(

	interface			=nio.FreeSurferSource(
	
		subjects_dir 		= subjects_dir),
	
			name		='fssource')
		

#Node: MRIConvert - to convert files from FreeSurfer format into nifti format
MRIconversion = pe.Node(

	interface			=fs.MRIConvert(
	
		out_type 		= 'nii'),
	
			name		='MRIconversion')


#Node: ApplyVolTransform - to transform contrasts into anatomical space
#                          creates 'con_*.anat.bb.mgh' files
transformation = pe.Node(

	interface			=fs.ApplyVolTransform(
	
		#fs_target		=True,
		#no_resample		=True),
		interp			='nearest'),
			name		='transformation')


#Node: SegStatsorig - to extract specified regions of the original segmentation
segmentationorig = pe.Node(

	interface			= fs.SegStats(
	
		segment_id 		= ROIregionsorig,
		#mask_thresh		= 0,
		mask_sign		= 'abs'),
		
			name		='segmentationorig')
			


#Node: SegStats2009 - to extract specified regions of the 2009 segmentation
segmentation2009 = pe.Node(

	interface			=fs.SegStats(
	
		segment_id		= ROIregions2009,
		mask_sign		= 'abs'),
		
			name		='segmentation2009')
			
			


#Node: Datasink - Creates a datasink node to store important outputs
datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = experiment_dir + '/results'
datasink.inputs.container = aROIOutput


#Connect up all components
aROIflow.connect([

		(inputnode, datagrabber,[('subject_id','subject_id'),
                                          ('contrast_id','contrast_id'),
                                          ]),
		(inputnode, transformation,[(
		
					(('subject_id',pathfinder),'target_file')
                                          )]),
                  (inputnode, fssource,[('subject_id','subject_id')]),
                  (fssource, segmentationorig,[(('aparc_aseg',getSegVersion,0),
                                                 'segmentation_file')]),
                  (fssource, segmentation2009,[(('aparc_aseg',getSegVersion,1),
                                                 'segmentation_file')]),
                 # (datagrabber, MRIconversion,[('contrast','in_file')]),
                  #(MRIconversion, transformation,[('out_file','source_file')]),
		  (datagrabber, transformation,	[('bb_id','reg_file')]),

                  (transformation, segmentationorig,[('transformed_file',
                                                      'in_file')]),
                  (transformation, segmentation2009,[('transformed_file',
                                                      'in_file')]),
                  (segmentationorig, datasink,[('summary_file', 'segstatorig')]),
                  (segmentation2009, datasink,[('summary_file', 'segstat2009')]),
                  ])
aROIflow.connect(datagrabber,'contrast',transformation, 'source_file')
aROIflow.write_graph(graph2use='flat')
aROIflow.run(plugin='MultiProc', plugin_args={'n_procs' : 12})


