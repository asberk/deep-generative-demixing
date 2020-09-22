#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000M
#SBATCH --time=00:30:00
#SBATCH --account=def-yaniv
#SBATCH --mail-user=aberk@math.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%u_%j_gd_cnn_vae_cifar_mnist.out
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "module load nixpkgs/16.09 gcc/7.3.0 arrow/0.11.1 python/3.6"
echo "module load scipy-stack opencv/3.4.3"
module load nixpkgs/16.09 gcc/7.3.0 arrow/0.11.1 python/3.6
module load scipy-stack opencv/3.4.3

echo ""
echo "source $HOME/RETINA/bin/activate"
source $HOME/RETINA/bin/activate

if [ $? -ne 0 ]
then
    echo "Failed to load environment."
else
    echo ""
    DATE=$(date "+%Y-%m-%d")
    echo "running script:"
    echo "  gd_numerics_two_network.py"
    echo "--data-class1=CIFAR10Subset --image-class1=2 --data-class2=MNISTSubset --image-class2=0 --network=CNN_VAE --network-latent-features=128 --network-device=cuda --optimizer-lr=1e-4"
    echo "($DATE)"
    echo ""
    python3 gd_numerics_two_network.py --data-class1=CIFAR10Subset --image-class1=2 --data-class2=MNISTSubset --image-class2=0 --network=CNN_VAE --network-latent-features=128 --network-device=cuda --optimizer-lr=1e-4
fi

echo "Complete."
