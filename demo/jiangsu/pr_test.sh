MY_Data_Name=jiangsu_pr
My_Model_Use=AB-MIL
MY_Classes=2
MY_Device=0
MY_Type=original

echo "----------training MIL model----------"
python /media/ipmi2022/704dca7c-f50a-410b-a2d3-9ae7944e09ff/syy/slideClassify/train_model.py \
--datasetsName $MY_Data_Name \
--n_classes $MY_Classes \
--model $My_Model_Use \
--k 0 \
--device $MY_Device \
--round 2 \
--lr 2e-4

