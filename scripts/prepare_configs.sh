#!/usr/bin/env bash

if [[ -z "$DBPORT" ]]; then
      DBPORT=$UseDBPORT
fi

if [[ -z "$LOGDIR" ]]; then
      LOGDIR="./logs"
fi

if [[ -z "$EXP_CONFIG" ]]; then
      EXP_CONFIG="./configs/config.yml"
fi

config_filename=$(basename -- "$EXP_CONFIG")
config_name="${config_filename%.*}"

date=$(date +%y%m%d-%H%M)
mkdir -p ${LOGDIR}/logs
mkdir -p ${LOGDIR}/${date}-mongodb


sed -i "s/dbPath: .*/dbPath: ${LOGDIR//\//\\/}\/$date-mongodb/g" ./configs/_mongod.conf
sed -i "s/path: .*/path: ${LOGDIR//\//\\/}\/$date-mongo.log/g" ./configs/_mongod.conf
sed -i "s/port: .*/port: ${DBPORT//\//\\/}/g" ./configs/_mongod.conf

sed -i "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/$date-$config_name/g" $EXP_CONFIG
sed -i "s/port: .*/port: ${DBPORT//\//\\/}/g" $EXP_CONFIG

export CUR_TB_LOGDIR=${LOGDIR}/logs/$date-$config_name
