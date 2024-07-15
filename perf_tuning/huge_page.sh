echo 32 | sudo tee /proc/sys/vm/nr_hugepages
cat /proc/meminfo | grep -i huge