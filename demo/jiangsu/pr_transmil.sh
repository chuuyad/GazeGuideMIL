MY_Data_Name=jiangsu_pr
My_Model_Use=Trans-MIL
MY_Classes=2
MY_Device=0

echo "----------training MIL model----------"
python train_model.py \
--datasetsName $MY_Data_Name \
--n_classes $MY_Classes \
--model $My_Model_Use \
--k 0 \
--device $MY_Device \
--round 0 \
--lr 2e-4
