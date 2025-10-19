for run in {1..1}
do
    seed=$((run + 40))
    echo "Outer loop iteration: $run (seed=$seed)"

    for sdiff_cons in 0.25
    do
        echo "NOTE: We first evaluate with apprx_iter=1, then purify again. Wait for the final results eval after adv perturbation ......"
        CUDA_VISIBLE_DEVICES="5" setsid torchrun --nproc_per_node=1 --master_port=27906 main_bpda_eot.py --dataset 'CIFAR10' --batch_size 8 \
            --network_pkl "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
            --clf_net 'wideresnet-28-10' --subset_size 512 --data_seed $seed --seed $seed \
            --purify_model 'opt' --forward_steps 0.25 --total_steps 1 --purify_iter 20 --apprx_iter 1 --ex_grad 1 \
            --att_method 'pgd_eot' --att_lp_norm -1 --att_eps 0.031373 --att_n_iter 20 --eot_attack_reps 20 \
            --lr 0.1 --purify_method 'aaopt' --sdiff_cons $sdiff_cons --aa 1 --sdiff_net "./pretrained_models/cifar10_score_DIFF_JOINT_wideresnet-28-10_"
        
        echo "Done with sdiff_cons=$sdiff_cons"
        echo " "
    done

    echo "Completed outer loop iteration: $run (seed=$seed)"
    echo "==============================="
done