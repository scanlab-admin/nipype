#!/bin/env python
import sys,os
import nipype.interfaces.freesurfer as fs # freesurfer
import nipype.interfaces.io as nio        # i/o routines
import nipype.interfaces.spm as spm       # spm
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe       # pypeline engine


#~~~~~~~~~~ user defined inputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

subject_list =['SAD_C19','SAD_C20','SAD_C21','SAD_C22','SAD_C23','SAD_C24','SAD_C25','SAD_C26','SAD_C27','SAD_C28','SAD_C29','SAD_C30','SAD_C31','SAD_C32','SAD_C33','SAD_C34','SAD_C35','SAD_C36','SAD_C37','SAD_C38','SAD_C39','SAD_C40','SAD_C41','SAD_C42','SAD_P04','SAD_P05','SAD_P06','SAD_P07','SAD_P08','SAD_P09','SAD_P11','SAD_P12','SAD_P13','SAD_P14','SAD_P15','SAD_P16','SAD_P17','SAD_P19','SAD_P20','SAD_P21','SAD_P22','SAD_P23','SAD_P24','SAD_P25','SAD_P26','SAD_P27','SAD_P28','SAD_P29','SAD_P30','SAD_P31']
subjects_dir 		= '/Volumes/lashley/DotPro.01/Analysis/nipype/'
study_FSGD		= subjects_dir + '/scripts/my.fsgd'
freesurfer_dir		= subjects_dir
numberOfContrasts 	= 39
cope_ids 		= range(1,numberOfContrasts+1)
contrast_ids 		= range(1,numberOfContrasts+1)
fs.FSCommand.set_default_subjects_dir(freesurfer_dir)

#SAD_C25 seems to be problematic.  Did not concatenate into the MRISpreproc step. 

#~~~~~~~~~~ groupwise contrasts ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

con_1 = ('Controls','T', ['Group_{1}'],[1])
con_2 = ('Patients','T', ['Group_{2}'],[1])
con_3 = ('Controls>Patients','T',['Group_{1}','Group_{2}'],[1,-1])
con_4 = ('Patients>Controls','T',['Group_{1}','Group_{2}'],[-1,1])
contrast_list = [con_1,con_2,con_3,con_4]


#~~~~~~~~~~ 3rd level volflow  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#initiate the 3rd level volume workflow
thirdlevel_volume = pe.Workflow(
	
	 			base_dir 		= subjects_dir,
	 			name			= "_3rd_level_VOL_results")


inputnode = pe.Node(
	
	
		interface=util.IdentityInterface(
		
			fields				=['contrasts']),
							
				name			= 'inputnode')

inputnode.inputs.contrasts 				= contrast_list


datagrabber = pe.Node(

		nio.DataGrabber(
		
			infields			=['con'],
			outfields			= ['control_copes', 'patient_copes'],),
	 
				name			="datagrabber")

datagrabber.inputs.template= '*'
datagrabber.inputs.field_template = dict(control_copes = subjects_dir + '/2nd_level_VOL_results/_subject_id_SAD_C*/_warp*/con_%04d_out_warped_wimt.nii',
				         patient_copes = subjects_dir + '/2nd_level_VOL_results/_subject_id_SAD_P*/_warp*/con_%04d_out_warped_wimt.nii',
					)
					
datagrabber.inputs.template_args = dict(control_copes	= [[ 'con' ]],
					patient_copes	= [[ 'con' ]],
					)	

datagrabber.iterables 					= [('con',cope_ids)] # iterate over all contrast images
datagrabber.inputs.sorted 				= True




level3_voldesign = pe.Node(
	
		interface 				= spm.TwoSampleTTestDesign(),
			
			name				= "level3_voldesign")


level3_volestimate = pe.Node(
	
		interface				= spm.EstimateModel(

			estimation_method 		= {'Classical' : 1}),
		   
				name			= "level3_volestimate")



level3_volcontrastestimate = pe.Node(
		
		interface 				= spm.EstimateContrast(
		 
			group_contrast			= True,),
						
				name			="level3_volcontrastestimate")





#Node: Threshold - to threshold the estimated contrast
level3_volthreshold = pe.MapNode(
		
		interface 				= spm.Threshold(
		
			contrast_index 			= 1,
			use_fwe_correction 		= True,
			#use_topo_fdr 			= True,
			force_activation		= True,
			extent_threshold 		= 1,
			extent_fdr_p_threshold 		= 0.05,
			height_threshold 		= 1.3),#cluster threshold (value is in -ln()): 1.301 = 0.05; 2 = 0.01; 3 = 0.001,
			iterfield			= ['stat_image'],
			name				="level3_volthreshold")


#connect up all volume components
thirdlevel_volume.connect(datagrabber,'control_copes'  ,level3_voldesign,'group1_files')
thirdlevel_volume.connect(datagrabber,'patient_copes'  ,level3_voldesign,'group2_files')
thirdlevel_volume.connect(inputnode,'contrasts',level3_volcontrastestimate,'contrasts')
thirdlevel_volume.connect(level3_voldesign,'spm_mat_file',level3_volestimate,'spm_mat_file')
thirdlevel_volume.connect(level3_volestimate,'spm_mat_file',level3_volcontrastestimate,'spm_mat_file')
thirdlevel_volume.connect(level3_volestimate,'beta_images',level3_volcontrastestimate,'beta_images')
thirdlevel_volume.connect(level3_volestimate,'residual_image',level3_volcontrastestimate,'residual_image')
thirdlevel_volume.connect(level3_volcontrastestimate,'spm_mat_file',level3_volthreshold,'spm_mat_file')
thirdlevel_volume.connect(level3_volcontrastestimate,'spmT_images',level3_volthreshold,'stat_image')


try:
	thirdlevel_volume.write_graph()
	thirdlevel_volume.run(plugin='MultiProc', plugin_args={'n_procs' : 12})
	print '====================================================\n'
	print "Workflow '3RD LEVEL ANALYSIS - VOLUME' Completed Normally.\n"
	print "====================================================\n"
	
except:
	print "====================================================\n"
	print "Workflow '3RD LEVEL ANALYSIS- VOLUME' Terminated with Errors.\n"
	print "====================================================\n"






#~~~~~~~~~~ 3rd level surf flow   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



thirdlevel_surface = pe.Workflow(
		
		  base_dir 				= subjects_dir,

				name			= "_3rd_level_SURF_results")
		
		
#Node: IdentityInterface - to iterate over contrasts and hemispheres
level3_surfinputnode = pe.Node(
		
		  interface				= util.IdentityInterface(
	
			fields				= ['con_id','hemi','subject_id']),
			
				name			= 'l3surfinputnode')

level3_surfinputnode.inputs.subject_id			= subject_list
level3_surfinputnode.iterables				= [('con_id', cope_ids),
							   ('hemi', ['lh','rh'])]


level3_surfdatagrabber = pe.Node(
		
		interface				= nio.DataGrabber(

			infields			= ['con_id'],
			outfields			= ['con_img','reg']),
			
				base_directory		= subjects_dir,
				name			= 'level3_surfdatagrabber')

level3_surfdatagrabber.inputs.template 			= '*'
level3_surfdatagrabber.inputs.field_template= dict(	con_img	=subjects_dir +'/2nd_level_SURF_results/_subject_id_*/_vol2vol*/con_%04d_out_warped.nii',
                                              		reg	=subjects_dir +'/2nd_level_SURF_results/_subject_id_*/_bbregister0/rf_dtype_out_st_bbreg_*.dat')
				
level3_surfdatagrabber.inputs.template_args = dict(	con_img	=[['con_id']],
							reg	=[[]])


def ordersubjects(files, subj_list):
	outlist = []
	for subj in subj_list:
		for file in files:	
			if '%s' %subj in file:
				outlist.append(file)
				continue
	return outlist

def list2tuple(listoflist):
	print [tuple(x) for x in listoflist]
	return [tuple(x) for x in listoflist]



level3_surfmerge = pe.Node(
		
		 interface				= util.Merge(2,
		
			axis				= 'hstack'),
			
				name			= 'level3_surfmerge')
			
			
concat = pe.Node(

		interface				= fs.MRISPreproc(
			
			target 				= 'avgsurf',
			fwhm 				= 5),
			
				name			= 'concat')


level3_surfTtest = pe.Node(

		interface				=fs.GLMFit(
		
		
			fsgd				= (study_FSGD,'dods'),
			contrast			= subjects_dir + '/scripts/BwGroupHCvSAD.mat',
			surf				= True,
			subject_id 			= 'avgsurf'),


				iterfield 		= ['hemi'],
				name			= 'level3_surfTtest')


'''
#Node: Datasink - Create a datasink node to store important outputs
thirdlevel_datasink = pe.Node(

		interface		= nio.DataSink(),
		
			name		= "l3datasink")
			
thirdlevel_datasink.inputs.base_directory 	= subjects_dir + '/3rd_level_results'
thirdlevel_datasink.inputs.container 		= nameOfLevel3Out
				
'''


			
thirdlevel_surface.connect(level3_surfinputnode,'con_id',level3_surfdatagrabber,'con_id')
thirdlevel_surface.connect(level3_surfinputnode,'hemi',level3_surfdatagrabber,'hemi')
thirdlevel_surface.connect([

                    (level3_surfdatagrabber,level3_surfmerge,[
					
					(('con_img', ordersubjects, subject_list),'in1'),
					(('reg', ordersubjects, subject_list),'in2'),

					])
					])

thirdlevel_surface.connect(level3_surfmerge,('out',list2tuple),concat,'vol_measure_file')
thirdlevel_surface.connect(level3_surfinputnode,'hemi',concat,'hemi') 
#thirdlevel_surface.connect(level3_surfinputnode,'subject_id',concat,'subjects') 


thirdlevel_surface.connect([		#((level3_surfdatagrabber,level3_surfTtest,[('subject_id','subject_id')])),

					((concat,level3_surfTtest,[('out_file','in_file')])),
					((level3_surfinputnode,level3_surfTtest,[('hemi','hemi')])),
				#	((level3_surfinputnode,level3_surfTtest,[('subject_id','subject_id')]))
				])

'''
#integration of the datasink into the volume analysis pipeline
l3surfflow.connect([(l3conestimate,l3datasink,[('spm_mat_file','l2vol_contrasts.@spm_mat'),
                                              ('spmT_images','l2vol_contrasts.@T'),
                                              ('con_images','l2vol_contrasts.@con'),
                                              ]),
                   (l3threshold,l3datasink,[('thresholded_map',
                                             'vol_contrasts_thresh.@threshold')]),
'''
#integration of the datasink into the surface analysis pipeline
#thirdlevel_surface.connect([(level3_surfTtest,l3datasink,[('sig_file','l3surf_contrasts.@sig_file')])])


try:
	thirdlevel_surface.write_graph()
	thirdlevel_surface.run(plugin='MultiProc', plugin_args={'n_procs' : 12})
	print '====================================================\n'
	print "Workflow '3RD LEVEL ANALYSIS - SURFACE' Completed Normally.\n"
	print "====================================================\n"
	
except:
	print "====================================================\n"
	print "Workflow '3RD LEVEL ANALYSIS- SURFACE' Terminated with Errors.\n"
	print "====================================================\n"

