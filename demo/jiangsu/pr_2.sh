MY_Data_Name=jiangsu_pr
My_Model_Use=AB-MIL
MY_Classes=2
MY_Device=0
MY_Type=original
MY_SlidePath=/media/ipmi2023-sc/1.44.1-42962/suyunyi/jiangsu_123

echo '----------extract feature----------'
python extract_feature_clean.py \
--datasetsName $MY_Data_Name \
--model $My_Model_Use \
--slide_dir $MY_SlidePath \
--device $MY_Device \
--round 2 \
--n_classes $MY_Classes

echo "----------training MIL model----------"
python train_model.py \
--datasetsName $MY_Data_Name \
--n_classes $MY_Classes \
--model $My_Model_Use \
--k 0 \
--device $MY_Device \
--round 2 \
--lr 2e-4


echo "----------pseudoLabeling{3}----------"
python PseudoLabeling.py \
--datasetsName $MY_Data_Name \
--model $My_Model_Use \
--n_classes $MY_Classes \
--round 3

echo "----------creating patches{3}----------"
python M1-1_create_patch.py \
--datasetsName $MY_Data_Name \
--model $My_Model_Use \
--slide_dir $MY_SlidePath \
--n_classes $MY_Classes \
--type $MY_Type \
--round 3

echo "----------update feature extraction{3}----------"
python M1-2_updata_model.py \
--datasetsName $MY_Data_Name \
--model $My_Model_Use \
--device $MY_Device \
--n_classes $MY_Classes \
--round 3

echo "----------extracting feature{3}----------"
python extract_feature_clean.py \
--datasetsName $MY_Data_Name \
--model $My_Model_Use \
--slide_dir $MY_SlidePath \
--device $MY_Device \
--n_classes $MY_Classes \
--round 3

echo "----------training MIL model{3}----------"
python train_model.py \
--datasetsName $MY_Data_Name \
--n_classes $MY_Classes \
--model $My_Model_Use \
--k 0 \
--device $MY_Device \
--round 3