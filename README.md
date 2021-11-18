# FEDn implementation of cifar10 with keras 

## Setting up a client


### Initiate your federation
This command loads and partitions the dataset, generates a 
docker-compose.yaml for your choice of numbers of clients and
initiates a seed model to start from:
(Replace: {NR_OF_CLIENTS} with the number of clients you want to build you federation with)
```bash
pip install -r init_requirements.txt
python init_federation.py {NR_OF_CLIENTS}
```


### Creating a compute package

```bash
tar -czvf package/kerascifar.tar.gz client
```


## Start the client
The easiest way to start clients for quick testing is by using Docker. We provide a docker-compose template for convenience. First, edit 'fedn-network.yaml' to provide information about the reducer endpoint. Then run following command in project directory:
(Replace {NR_OF_CLIENTS} with the number of clients you selected)
```bash
docker-compose -f docker-compose-{NR_OF_CLIENTS}clients.yaml up 
```

> Note that this assumes that a FEDn network is running with "keras" helper, which is identified in "config/settings-reducer.yaml" (see separate deployment instructions). The file 'docker-compose.dev.yaml' is for testing againts a local pseudo-distributed FEDn network. Use 'docker-compose.yaml' if you are connecting against a reducer part of a distributed setup and provide a 'extra_hosts' file.
