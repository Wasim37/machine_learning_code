#高评分电影的主演词云
from wordcloud import WordCloud,ImageColorGenerator
from scipy.misc import imread
import matplotlib.pyplot as plt
import pymysql

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

conn=pymysql.connect(host='localhost',user='root',passwd='123456',db='movie',charset='utf8')
cur=conn.cursor()
sql='select actor from maoyan'
cur.execute(sql)
actor_list=cur.fetchall()
print(actor_list)
actor_name_list=[item[0] for item in actor_list]
actor_text=' '.join(actor_name_list)

#设置背景图片
coloring = imread('girl.png')

font='C:\Windows\Fonts\simkai.ttf'
#实例化词云
my_wordcloud=WordCloud(font_path=font,max_font_size=300,background_color='white',mask=coloring).generate(actor_text)

#从背景颜色生成颜色值
image_colors=ImageColorGenerator(coloring)
plt.imshow(my_wordcloud.recolor(color_func=image_colors))
plt.axis('off')
plt.show()