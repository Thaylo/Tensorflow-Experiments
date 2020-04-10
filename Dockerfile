FROM intelaipg/intel-optimized-tensorflow
ADD main.py /
RUN pip3 install pandas
RUN pip3 install scikit-learn
CMD [ "python", "./main.py" ]