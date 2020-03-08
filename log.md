### 20/03/06
开始撸代码了！先了解一下movieLens数据集的内容~

* tags.csv
  
    userId<br>
    movieId<br>
    tag：由用户决定<br>
    timestamp：自1970年1月1日零点后到用户提交评价的时间的秒数<br>

* ratings.csv
  userId<br>
  movieId<br>
  rating：0.5~5.0<br>
  timestamp<br>

* movies.csv
  movieId:193609<br>
  title<br>
  genres：给定的分类，一部影片可以跨越多个分类<br>

* links.csv
  movieId<br>
  imdbId：应该是在imdb这个网站中的id<br>
  tmdbId：themoviedb上的id<br>
  上面这俩都可以在网站中找到<br>



con：好像也没干啥，弄清楚了dataset里面都有啥。然后有了前车之鉴，这一次搭的虚环境。<br>
TODO
* 从csv里面读文件啦
* 弄随机抽取用户的函数（总共610个用户）感觉有点少，要不直接作为一个区域用户集⑧

### 2020/03/07
* 读文件的问题解决了
* ahp算法在写了

TODO
* 考虑清楚到底是否需要把数据二八分用作训练和验证，还是说只得到推荐的内容就好
* 时间戳要不要转换一下？yes的话怎么转换？


### 2020/03/08
* 其实读文件内一步应该改进一下的，不读空行的话严谨一些(算了没大碍)
* 评论，点击，评分的统计都整完了，直接放在了listMovie的全局变量里，现在想来是不是用numpy数组会好一些？


TODO
* 虽然ahp算法能照葫芦画瓢搬下来了，可是完全没懂为什么啊！我琢磨着你就是不愿意动脑子！！！**要搞明白哦，万一复试问到了？？**
* 明天争取把ahp部分都理顺咯