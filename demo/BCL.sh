MY_Data_Name=
My_Model_Use=AB-MIL
MY_Classes=
MY_Device=1
MY_Type=original
MY_SlidePath=

echo '----------extract feature{0}----------'
python extract_feature_clean.py \
--datasetsName $MY_Data_Name \
--model $My_Model_Use \
--slide_dir $MY_SlidePath \
--batch_size 256 \
--workers 8 \
--device $MY_Device \
--round 0 \
--n_classes $MY_Classes

echo "----------training MIL model{0}----------"
python train_model.py \
--datasetsName $MY_Data_Name \
--n_classes $MY_Classes \
--model $My_Model_Use \
--k 0 \
--device $MY_Device \
--round 0 \
--lr 2e-4

iteration=(1 2 3)
for num in "${iteration[@]}"
do
    echo "----------pseudoLabeling{$num}----------"
    python PseudoLabeling.py \
    --datasetsName $MY_Data_Name \
    --model $My_Model_Use \
    --n_classes $MY_Classes \
    --round $num

    echo "----------creating patches{$num}----------"
    python M1-1_create_patch.py \
    --datasetsName $MY_Data_Name \
    --model $My_Model_Use \
    --slide_dir $MY_SlidePath \
    --n_classes $MY_Classes \
    --type $MY_Type \
    --round $num

    echo "----------update feature extraction{$num}----------"
    python M1-2_updata_model.py \
    --datasetsName $MY_Data_Name \
    --model $My_Model_Use \
    --device $MY_Device \
    --n_classes $MY_Classes \
    --round $num

    echo "----------extracting feature{$num}----------"
    python extract_feature_clean.py \
    --datasetsName $MY_Data_Name \
    --model $My_Model_Use \
    --slide_dir $MY_SlidePath \
    --device $MY_Device \
    --n_classes $MY_Classes \
    --round $num

    echo "----------training MIL model{$num}----------"
    python train_model.py \
    --datasetsName $MY_Data_Name \
    --n_classes $MY_Classes \
    --model $My_Model_Use \
    --k 0 \
    --device $MY_Device \
    --round $num
done