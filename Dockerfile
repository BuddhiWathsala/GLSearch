FROM nvcr.io/nvidia/pytorch:19.04-py3


RUN pip install certifi==2019.3.9 \
chardet==3.0.4 \
cycler==0.10.0 \
decorator==4.4.0 \
dill==0.2.9 \
idna==2.8 \
isodate==0.6.0 \
kiwisolver==1.0.1 \
klepto==0.1.6 \
matplotlib==3.0.3 \
networkx==2.2 \
numpy==1.16.2 \
pandas==0.24.2 \
Pillow==6.0.0 \
plyfile==0.7 \
pox==0.2.5 \
protobuf==3.7.1 \
pyparsing==2.3.1 \
python-dateutil==2.8.0 \
pytz==2018.9 \
rdflib==4.2.2 \
requests==2.21.0 \
scikit-learn==0.20.3 \
scipy==1.2.1 \
six==1.12.0 \
tensorboardX==1.6 \
torchvision==0.2.2.post3 \
torch-cluster==1.2.4 \
torch-geometric==1.1.2 \
torch-scatter==1.1.2 \
torch-sparse==0.2.4 \
torch-spline-conv==1.0.6 \
tqdm==4.31.1 \
urllib3==1.24.1



#RUN wget https://web.cs.dal.ca/~peter/software/pynauty/pynauty-0.6.0.tar.gz && tar -vxf pynauty-0.6.0.tar.gz

#RUN cd pynauty-0.6.0 && wget http://users.cecs.anu.edu.au/~bdm/nauty/nauty25r9.tar.gz && tar -vxf nauty25r9.tar.gz && ln -s nauty25r9 nauty

#RUN cd pynauty-0.6.0 && make pynauty && python setup.py install

RUN pip install colour==0.1.5  \
shapely==1.7.0 \
comet-ml==3.1.6

WORKDIR /workspace
RUN git clone https://github.com/DerekQXu/GLSearch.git
RUN mv GLSearch/* .
COPY load/ ./model/OurMCS
WORKDIR /workspace/model/OurMCS/
CMD [ "python", "test.py" ]

