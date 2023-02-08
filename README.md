# Federated_Microgrid
Secure Federated Learning for predicting Smart Grid Stability


Git clone the package
```
git clone https://github.com/anwaybose/Federated_Microgrid.git
```

Install flower and tensorflow
```
python3 -m pip install flwr tensorflow-cpu
```

Genereate the certificates
```
cd Federated_Microgrid/certificates
sudo bash generate.sh
```

Open a terminal to run server
```
cd Federated_Microgrid
sudo python3 microgrid_federated_server.py
```

Open client terminals
```
cd Federated_Microgrid
python3 microgrid_federated_client.py   
```
