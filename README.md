# Test Suite for NN Surrogate B3LYP XC Funtional

## Workflow

1. Use Psi4 to calculate moleule of interest at B3LYP/aug-cc-pvtz level of theory, and project the SCF converged density and XC energy density on a grid.


`python psi4_generate_data_set.py <database for molecule coord> <molecule name> <h> <L> <N>`


	* <database for molecule coord>: provided, the `molecule_coord_database.json`. if you wish to test additional systems that's not defined in this file, please follow the same format
	* <molecule name>: name of the molecule as defined in the database
	* <h>: grid spacing in angstrom. In our case, please keep it as **0.02** 
	* <L>: overall dimension of the cubic box in angstrom. Please make sure there is engough vacuum padding. **10.0** is a good choice for the small systems
	* <N>: number of segmentation in each dimemsion for domain decomposition (generating sub grid for easier manipulation). for the paper, we used **5**.

	The command will generate the projected SCF converged density and XC energy density and save them as ".hdf5" files in `/molecule_database/<molecule>_B3LYP_<L>_<h>_<N>`

2. Calculate the MCSH convolutional descriptors.


`python getDescriptors_sec.py <setup filename> <molecule name>`


	* <setup filename>: setup file for calculating descriptors. a sample file is provided at `/setup_files/10-0_0-02_5_MCSH_descriptor_generation_setup.json`
	* <molecule name>: name of the molecule as defined in the database

	The script will take the files generated in previous step and calculate the MCSH descriptors of 0, 1, 2 orders and save the results again to ".hdf5" files 

3. Predict the energy with the model provided.


`python predict_model.py <predict setup> <descriptor set definition database> <descriptor set name>`


	* <predict setup>: setup for the prediction. An example can be found at `/setup_files/10-0_0-02_5_predict_100_2_relu_setup_LDAres_1M_setup.json`. Please modify the setup file for the right **database path**, **box dimension settings** and **list of molecules**
	* <descriptor set definition database>: provided, "descriptor_set_database.json"
	* <descriptor set name>: models for 4 desciptor sets are provided, so please choose between them: "epxc_MCSH_1_0-08_real_real", "epxc_MCSH_1_0-08_real_real", "epxc_MCSH_1_0-20_real_real", "epxc_MCSH_2_0-08_real_real","epxc_MCSH_1_0-20_real_real"

	this command will predict the XC energy with the models, and the result can be found in the log files generated in the model directory "/models/<descirptor set name>/<NN_LDA_residual_1M_100_2_relu>":

	- "predict_log.log": result for each molecular system
	- "predict_error_log.log": result for each molecular system in tsv format `<molecule>	<original xc>	<predicted xc>	<prediction error>	<sum of absolute error>`
	- "predict_full_log.log": detailed result with prediction error of each sub-grid

	if you have "NH3", "H2O", "H2", "CH4" systems listed in the moleulce_list in the setup file, formation energies can also be calculated. 

	- "predict_formation_log.log": result for each molecular system in tsv format `<molecule>	<original xc>	<predicted xc>	<prediction error>	<sum of absolute error>		<original xc formation>	<predicted xc formation>	<xc formation prediction error>`

	you can also change the reference systems at line 725 of `predict_model.py` 
	atomic_references = {'N':'NH3','O':'H2O','H':'H2','C':'CH4'}


## Dependencies

* [psi4 numpy](https://github.com/psi4/psi4numpy)
* numpy
* scipy
* h5py

