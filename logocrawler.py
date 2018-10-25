import logging
import os.path as osp
import shutil
import time
import base64
from icrawler import ImageDownloader
from icrawler.builtin import GoogleImageCrawler
try:
    from urllib.parse import urlparse
except ImportError:
     from urlparse import urlparse

from icrawler.builtin import (BaiduImageCrawler, BingImageCrawler,
                              GoogleImageCrawler, GreedyImageCrawler,
                              UrlListCrawler)
import argparse
#'361°logo','361°公司logo','361度衣服logo','361度鞋子logo','361度品牌logo','361度裤子logo'
import os
parser = argparse.ArgumentParser(description='Crawler')
parser.add_argument('--logo', default=['sensetime','商汤科技logo'], type=list,
                    help='input the logo name')
parser.add_argument('--maxnum', default=300, type=int,
                    help='the number to crawler')
args = parser.parse_args()
root = 'logo'
root = os.path.join(root,args.logo[0])
if not os.path.exists(root):
    os.mkdir(root)
"""Unit test is expected to be here, while we use some usage cases instead."""

class MyImageDownloader(ImageDownloader):

    def download(self,
                 task,
                 default_ext,
                 timeout=5,
                 max_retry=3,
                 overwrite=False,
                 **kwargs):
        """Download the image and save it to the corresponding path.

        Args:
            task (dict): The task dict got from ``task_queue``.
            timeout (int): Timeout of making requests for downloading images.
            max_retry (int): the max retry times if the request fails.
            **kwargs: reserved arguments for overriding.
        """
        file_url = task['file_url']
        task['success'] = False
        task['filename'] = None
        retry = max_retry
        fs = open(kwargs['filename'],'a+')
        del kwargs['filename']
        if not overwrite:
            with self.lock:
                self.fetched_num += 1
                filename = self.get_filename(task, default_ext)
                if self.storage.exists(filename):
                    self.logger.info('skip downloading file %s', filename)
                    return
                self.fetched_num -= 1

        while retry > 0 and not self.signal.get('reach_max_num'):
            try:
                response = self.session.get(file_url, timeout=timeout)
            except Exception as e:
                self.logger.error('Exception caught when downloading file %s, '
                                  'error: %s, remaining retry times: %d',
                                  file_url, e, retry - 1)
            else:
                if self.reach_max_num():
                    self.signal.set(reach_max_num=True)
                    break
                elif response.status_code != 200:
                    self.logger.error('Response status code %d, file %s',
                                      response.status_code, file_url)
                    break
                elif not self.keep_file(task, response, **kwargs):
                    break
                with self.lock:
                    self.fetched_num += 1
                    filename = self.get_filename(task, default_ext)
                self.logger.info('image #%s\t%s', self.fetched_num, file_url)
                fs.write('{:06d},{}\n'.format(self.fetched_num, file_url))
                self.storage.write(filename, response.content)
                task['success'] = True
                task['filename'] = filename
                break
            finally:
                retry -= 1
        
def test_google(logo):
    google_crawler = GoogleImageCrawler(
        downloader_cls=MyImageDownloader,
        downloader_threads=4,
        storage={'root_dir': os.path.join(root,logo,'google')},
        log_level=logging.INFO,
        filename=os.path.join(root,logo,'google.txt'))
    google_crawler.crawl(logo, max_num=args.maxnum)

def test_bing(logo):
    bing_crawler = BingImageCrawler(
        downloader_cls=MyImageDownloader,
        downloader_threads=4,
        storage={'root_dir': os.path.join(root,logo,'bing')},
        log_level=logging.INFO,
        filename=os.path.join(root,logo,'bing.txt'))
    bing_crawler.crawl(logo, max_num=args.maxnum)

def test_baidu(logo):
    baidu_crawler = BaiduImageCrawler(downloader_cls=MyImageDownloader,
        downloader_threads=4, storage={'root_dir': os.path.join(root,logo,'baidu')},
        log_level=logging.INFO,filename=os.path.join(root,logo,'baidu.txt'))
    baidu_crawler.crawl(logo, max_num=args.maxnum)

def test():
    s = time.time()
    for logo in args.logo:
        if not os.path.exists(os.path.join(root,logo)):
            os.mkdir(os.path.join(root,logo))
        test_google(logo)
        test_bing(logo)
        test_baidu(logo)
    time_elapsed = time.time() - s
    print('the time of one logo: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
test()