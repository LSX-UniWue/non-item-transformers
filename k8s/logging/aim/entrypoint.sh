# initialize aim storage if necessary
[ ! -d "/aim/.aim" ] && aim init --repo /aim

# determine mode and start grpc server or web server
if [ "${AIM_MODE}" == "grpc" ]; then
  echo "Starting grpc server"
  aim server --host 0.0.0.0 --port 53800 --workers 2 --repo /aim
elif [ "${AIM_MODE}" == "web" ]; then
  echo "Starting web server"
  aim up --host 0.0.0.0 --port 43800 --workers 2 --repo /aim
else
  echo "Did not recognize command."
fi


