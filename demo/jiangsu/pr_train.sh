MY_Data_Name=jiangsu_pr
My_Model_Use=AB-MIL
MY_Classes=2
MY_Device=0
MY_Type=original
MY_SlidePath=/media/ipmi2023-sc/1.44.1-42962/suyunyi/jiangsu_123

echo "----------training MIL model----------"
python /home/ipmi2023-sc/PycharmProjects/SlideClassify_stardard20240310/train_model.py \
--datasetsName $MY_Data_Name \
--n_classes $MY_Classes \
--model $My_Model_Use \
--k 0 \
--device $MY_Device \
--round 2 \
--lr 2e-4

