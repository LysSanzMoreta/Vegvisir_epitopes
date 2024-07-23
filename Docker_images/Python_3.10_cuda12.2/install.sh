docker image rm python39 --force #remove image completely
#docker rm $(docker ps -a | grep -v "pycharm" | awk 'NR>1 {print $1}') #remove old containers except pycharm
docker rm -v -f $(docker ps -qa)
docker builder prune #delete cache
docker rmi $(docker images -f "dangling=true" -q) #remove untagged/uncompleted images
docker build --no-cache --tag python39 . #build new image with same name ignoring cache

