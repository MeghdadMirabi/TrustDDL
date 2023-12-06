# TrustDDL: Towards a Privacy-Preserving Byzantine-Robust Distributed Deep Learning Framework

TrustDDL, a distributed deep learning framework designed to address privacy and Byzantine robustness concerns throughout 
the training and inference phases of deep learning models. TrustDDL employs additive secret-sharing-based protocols, a 
commitment phase, and redundant computation to detect Byzantine parties and safeguard the system from their adverse effects 
during both model training and inference. It guarantees uninterrupted protocol execution, ensuring reliable output delivery 
in both model training and inference phases.

## Docker Installation (Recommended)

We have executed our experiments within Docker containers based on the <code>ubuntu:22.04</code> image. For that, we 
provide a <code>Dockerfile</code>. You may simply create the image by executing the commands

```
git clone https://github.com/DataSecPrivacyLab/TrustDDL.git TrustDDL
cd TrustDDL
docker build -t trustddl .
```

and run the container by executing the command

```
docker run -it --net=host --privileged --shm-size=10.24gb --name trustddl trustddl '/bin/bash'
```

You may detach from the interactive shell by pressing <code>Ctrl+p</code>, <code>Ctrl+q</code> and re-attach to the shell 
by executing the command <code>docker attach trustddl</code>.

## Manual Installation

You may also install this project manually by running the commands

```
git clone https://github.com/DataSecPrivacyLab/TrustDDL.git TrustDDL
cd TrustDDL
pip install -r TrustDDL/requirements.txt
```

These commands require an installation of <code>python3</code> and <code>pip</code>. We advise against a manual installation
for LAN / WAN setups if it cannot be guaranteed that the same python version will be used on all servers. 

## Experiment Parameters

In the file <code>configuration.py</code> you can configure the following parameters:

> - **network**: The network to be used in the experiment. The current implementation supports the networks 'SecureML'
    and 'Chameleon'. For each network (and batch size) we configure default learning rates in the file <code>util.constants.py</code>.
> - **batch_size**: The batch size to be used in the experiment.
> - **threat_model**: The threat model to be used in the experiment. Accepted values for this parameter are 'semi-honest'
    and 'malicious'.
> - **threads_available**: The number of threads to be used for the execution of the experiment. You may set this to twice
    the number of available cores or use it to limit resource utilization.  
> - **epochs**: The number of epochs to be used in experiments regarding the **accuracy** of neural networks in our framework.
    This parameter will be used exclusively in the <code>eval_accuracy.py</code> script.
> - **iterations**: The number of iterations to be used in experiments regarding the **runtime** or **communication cost**
    of neural networks in our framework. This parameter will be used in the scripts <code>eval_runtime.py</code> and 
    <code>eval_comm.py</code>.
> - **train**: Whether to perform training (<code>True</code>) or inference tasks (<code>False</code>) in the experiments
    regarding the **runtime** or **communication cost**.
> - **log_warnings**: Debugging parameter used to enable (<code>True</code>) or disable (<code>False</code>) warnings issued 
    by the Ray framework. Enabling this parameter might give some insight into problems regarding the setup of the servers. 

## Single Machine Setup

On a single machine, you may simply run the experiments by configuring the desired parameters and executing the command

```
python3 ${evaluation_script}
```

(replace <code>${evaluation_script}</code> with the desired evaluation script; the provided evaluation scripts are listed below). We 
evaluate our framework solely on the MNIST data set. In this project, we provide the following evaluation scripts:

> - <code>eval_accuracy.py</code>: A script to evaluate the accuracy of neural networks trained in our framework. By default,
    the accuracy results are evaluated after every 10,000 iterations and printed to the screen. Furthermore, the results 
    are written to a file <code>results/${network}\_epochs\_${epochs}\_bs_${batch_size}.txt</code>.
>- <code>eval_pretrained_accuracy.py</code>: A script to evaluate the accuracy of neural networks trained in the PyTorch 
    framework and deployed in our framework. We print the accuracy of the network on the MNIST test set in a centralized 
    deployment and for the deployment in our framework.
> - <code>eval_runtime.py</code>: A script to evaluate the runtime of neural networks in our framework. We print the complete
    runtime for all iterations, the runtime of each iteration, the average runtime of all iterations and the average runtime
    of all iterations except the first. The last output is given due to the fact that the first iteration is always significantly
    slower than all other iterations due to the setup time in Ray.
> - <code>eval_comm.py</code>: A script to evaluate the communication cost of neural networks in our framework. Here, we
    print the total number of messages sent and the overall network traffic incurred by the training / inference task(s).

## LAN / WAN Setup

To run our experiments in a LAN or WAN setup, we will create a Ray cluster and deploy our actors over this cluster. For 
that, we will first choose one of the servers as the *head node* of the cluster. On this server we will execute the command

```
ray start --head
```

To limit the number of threads to be used, we may also specify an optional resource parameter. In our evaluation, we have 
used one server for each party. To start a party _i=0,1,2_, we will need to specify the resources
<code>--resources='{"cpu": ${threads}, "party${i}": 1}</code> on server _i_
(replace <code>${threads}</code> with the number of threads to use and <code>${i}</code> with the party / server number out of {0,1,2}).
You may remove the <code>"cpu"</code> resource if you want to use all available threads on a server.

To start the other servers, you may execute the command

```
ray start --address='${ip_head_node}:${assigned_ray_port}'
```

(replace <code>${ip_head_node}</code> with the IP of the head node and <code>${assigned_ray_port}</code> with the port
assigned at the startup of the head node (will be printed with the <code>ray start</code> command)). Again, we will need
to specify the group resources and optionally limit the number of threads with the resource parameter with the resource 
parameter given above.

To place the Data Owner and Model Owner on any given server, we may specify the resource <code>"Owners": 1</code>
in the Ray start commands.

**Example**: We may start four servers (one for each party + one for the Data Owner and Model Owner), we may execute the commands

```
ray start --head --resources='{"Owners": 1}
ray start --address='${ip_head_node}:${assigned_ray_port}' --resources='{"party0": 1}
ray start --address='${ip_head_node}:${assigned_ray_port}' --resources='{"party1": 1}
ray start --address='${ip_head_node}:${assigned_ray_port}' --resources='{"party2": 1}
```

Finally, you may start any of the evaluation scripts by running the command

```
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit -- python3 ${evaluation_script}
```

on the head node (replace <code>${evaluation_script}</code> with the desired evaluation script).