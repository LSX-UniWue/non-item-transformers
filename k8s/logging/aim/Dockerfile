FROM python:3.9

# install the `aim` package on the latest version
RUN pip install --upgrade aim==3.6.3

# grpc or web
ENV AIM_MODE "grpc"

# make a directory where the Aim repo will be initialized, `/aim`
RUN mkdir /aim
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
