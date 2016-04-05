FROM ubuntu:15.10
ENV DEBIAN_FRONTEND noninteractive

RUN echo "UTC" > /etc/timezone
RUN dpkg-reconfigure tzdata

# Set locale
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LANG C.UTF-8

# create deploy user
RUN useradd --create-home --home /var/lib/deploy deploy

# install apt-get requirements
ADD apt-requirements.txt /tmp/apt-requirements.txt
RUN apt-get update -y
RUN xargs -a /tmp/apt-requirements.txt apt-get install -y

# install Java JDK 8
RUN sed 's/main$/main universe/' -i /etc/apt/sources.list
RUN add-apt-repository ppa:webupd8team/java -y
RUN apt-get update
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections
RUN apt-get install -y oracle-java8-installer zip

# download & install leiningen
RUN su deploy -c 'mkdir /var/lib/deploy/bin'
RUN su deploy -c 'curl -L "https://raw.github.com/technomancy/leiningen/stable/bin/lein" -o /var/lib/deploy/bin/lein'
RUN chmod +x /var/lib/deploy/bin/lein
RUN su - deploy -c 'lein upgrade'

# download and install zmq
RUN add-apt-repository ppa:chris-lea/zeromq
RUN add-apt-repository ppa:chris-lea/libpgm
RUN apt-get update
RUN apt-get install -y libzmq3-dev

# node.js and utils
RUN add-apt-repository ppa:chris-lea/node.js
RUN apt-get install -y nodejs npm && npm update
ENV NODE_PATH $NODE_PATH:/usr/local/lib/node_modules
RUN npm install -g requirejs
RUN ln -s /usr/bin/nodejs /usr/bin/node

# node.js deps
RUN npm install q underscore zmq atob commander


## From here on we're the deploy user
USER deploy
# install Anaconda
RUN aria2c -s 16 -x 16 -k 30M https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda3-4.0.0-Linux-x86_64.sh -o /var/lib/deploy/Anaconda.sh
RUN cd /var/lib/deploy && bash Anaconda.sh -b && rm -rf Anaconda.sh
ENV PATH=/var/lib/deploy/anaconda3/bin:$PATH

# install Python dependencies
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# NLTK stuff
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords

# Get the source
RUN mkdir /var/lib/deploy
ADD deploy.tar.gz /var/lib/deploy
RUN mkdir /var/lib/deploy/src/
RUN tar -xzf archive.tar -C /var/lib/deploy/src/

# compile client side assets
RUN su - deploy -c 'cd /var/lib/deploy/src/resources &&  r.js -o public/build.js && rm -rf public && mv build public'

# add run file
ADD run.sh /var/lib/deploy/bin/run
RUN chown deploy.deploy /var/lib/deploy/bin/run
RUN su - deploy -c 'chmod +x bin/run'

EXPOSE 8888
USER deploy
ENV HOME /var/lib/deploy
ENV DEV false
ENTRYPOINT ["/var/lib/deploy/bin/run"]
CMD ["run", "start"]