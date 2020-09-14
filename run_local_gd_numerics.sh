echo ""
DATE=$(date "+%Y-%m-%d")
echo "running script:"
echo "  gd_numerics_two_network.py"
echo "($DATE)"
echo ""
python gd_numerics_two_network.py --network=CNN_VAE --network-latent-features=128 --network-device=cpu
echo "Complete."

