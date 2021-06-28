import glob
import os

from PIL import Image
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from flask import Flask, render_template, request, Response, redirect
from werkzeug.utils import secure_filename
import json
from feature import FeatureExtractor
from pathlib import Path
import numpy as np

from capgen import CaptionGenerator

import cv2
import uuid
from annoy import AnnoyIndex
import redis
import hashlib

os.environ['CUDA_VISIBLE_DEVICES'] = ''
es = Elasticsearch(hosts='e.ipipip.com', port=6001)
gencap = CaptionGenerator()

sift = cv2.SIFT_create()
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))


def get_match_num(matches, ratio):
    matches_mask = [[0, 0] for _ in range(len(matches))]
    match_num = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:  # 将距离比率小于ratio的匹配点删选出来
            matches_mask[i] = [1, 0]
            match_num += 1
    return match_num, matches_mask


def description_search(query):
    global es
    results = es.search(
        index="desearch",
        body={
            "size": 20,
            "query": {
                "match": {"description": query}
            }
        })
    hitCount = results['hits']['total']
    print(results)

    if hitCount > 0:
        if hitCount is 1:
            print(str(hitCount), ' result')
        else:
            print(str(hitCount), 'results')
        answers = []
        max_score = results['hits']['max_score']

        if max_score >= 0.35:
            for hit in results['hits']['hits']:
                if hit['_score'] > 0.5 * max_score:
                    desc = hit['_source']['description']
                    imgurl = hit['_source']['imgurl']
                    answers.append([imgurl, desc])
    else:
        answers = []
    return answers


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'database')
app.config['TEMP_UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('search.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        if 'query_img' not in request.files or request.files['query_img'].filename == '' or not allowed_file(
                request.files['query_img'].filename):
            return render_template('search.html')
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = os.path.join(app.config['TEMP_UPLOAD_FOLDER'], file.filename)
        img.save(uploaded_img_path)

        fe = FeatureExtractor()
        query = fe.extract(img=img)

        features = []
        img_paths = []
        r = redis.StrictRedis(host='redis', decode_responses=True)
        u = AnnoyIndex(4096, 'angular')
        if os.path.exists('feature.ann'):
            u.load('feature.ann')
            all_vectors = u.get_nns_by_vector(query.ravel(), 10)
            for i in all_vectors:
                feature = np.array(u.get_item_vector(i), dtype='float32')
                features.append(feature)
                key = hashlib.sha1(feature).hexdigest()
                img_paths.append(Path("./static/database") / r.get(key))

        # for feature_path in Path("./static/feature").glob("*.npy"):
        #     features.append(np.load(str(feature_path)))
        #     img_paths.append(Path("./static/database") / feature_path.stem)
        features = np.array(features)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features

        ids = np.argsort(dists)[:30]
        answers = [(img_paths[id], round(dists[id],2), '') for id in ids]

        good = [x for x in answers if x[1] < 1.1]
        bad = [x for x in answers if x[1] >= 1.1]
        return render_template('search.html',
                               query_path=uploaded_img_path,
                               answers=good,
                               answers2=bad)
    else:
        return render_template('search.html')


@app.route('/siftsearch', methods=['GET', 'POST'])
def sift_search():
    if request.method == 'POST':
        if 'query_img' not in request.files or request.files['query_img'].filename == '' or not allowed_file(
                request.files['query_img'].filename):
            return render_template('search.html')
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = os.path.join(app.config['TEMP_UPLOAD_FOLDER'], file.filename)
        img.save(uploaded_img_path)

        img1 = cv2.imread(uploaded_img_path, 0)
        img1 = cv2.resize(img1, (224, 224))
        kp1, des1 = sift.detectAndCompute(img1, None)  # 提取比对图片的特征
        answers = []
        # images = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
        # for image in images:
        #     img2 = cv2.imread(image, 0)
        #     kp2, des2 = sift.detectAndCompute(img2, None)
        #     matches = matcher.knnMatch(des1, des2, k=2)  # 匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
        #     (match_num, matches_mask) = get_match_num(matches, 0.9)  # 通过比率条件，计算出匹配程度
        #     match_ratio = match_num * 100 / len(matches)
        #     draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matches_mask, flags=0)
        #     comparison_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        #     tmp_file_name = "./static/tmp/" + uuid.uuid4().hex + ".jpg"
        #     cv2.imwrite(tmp_file_name, comparison_image)
        #     sift_answers.append((tmp_file_name, match_ratio))

        # for feature_path in Path("./static/sift").glob("*.npy"):
        #     des2 = np.load(str(feature_path))
        #     matches = matcher.knnMatch(des1, des2, k=2)  # 匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
        #     (match_num, matches_mask) = get_match_num(matches, 0.9)  # 通过比率条件，计算出匹配程度
        #     match_ratio = match_num * 100 / len(matches)
        #     answers.append((Path("./static/database") / feature_path.stem, match_ratio))

        r = redis.StrictRedis(host='redis', decode_responses=True)
        feature_size = 128 * 128
        u = AnnoyIndex(feature_size, 'angular')
        if os.path.exists('feature_sift.ann'):
            u.load('feature_sift.ann')
            des1 = des1.flatten()
            if des1.size < feature_size:
                data_size_1 = des1.size
                des1 = np.concatenate([des1, np.zeros(feature_size - des1.size, dtype='float32')])
            else:
                data_size_1 = feature_size

            des1 = des1[:feature_size]

            all_vectors = u.get_nns_by_vector(des1, 10)
            des1 = des1[:data_size_1]
            feature_count_1 = int(data_size_1 / 128)
            kp1_len = ' / ' + str(len(kp1))
            for i in all_vectors:
                des2 = np.array(u.get_item_vector(i), dtype='float32')
                key = hashlib.sha1(des2).hexdigest()
                data_size_2 = r.get(key + '_size')
                if data_size_2:
                    feature_count_2 = int(int(data_size_2) / 128)
                    des2 = des2[:int(data_size_2)]
                else:
                    feature_count_2 = 128
                matches = matcher.knnMatch(des1.reshape(feature_count_1, 128), des2.reshape(feature_count_2, 128), k=2)  # 匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
                (match_num, matches_mask) = get_match_num(matches, 0.9)  # 通过比率条件，计算出匹配程度
                match_ratio = match_num * 100 / len(matches)
                # answers.append((Path("./static/database") / r.get(key), round(match_ratio, 2)))
                answers.append((Path("./static/database") / ('f_' + r.get(key)), match_num, '/ ' + str(len(matches)) + kp1_len))

        answers.sort(key=lambda x: x[1], reverse=True)  # 按照匹配度排序
        good = [x for x in answers if x[1] > 50]
        bad = [x for x in answers if x[1] <= 50]

        return render_template('search.html',
                               query_path=uploaded_img_path,
                               answers=good,
                               answers2=bad,
                               action='sift')
    else:
        return render_template('search.html', action='sift')


@app.route('/search2', methods=['GET', 'POST'])
def search2():
    global gencap
    if request.method == 'POST':
        if 'query_img' not in request.files or request.files['query_img'].filename == '' or not allowed_file(
                request.files['query_img'].filename):
            return render_template('search.html')
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = os.path.join(app.config['TEMP_UPLOAD_FOLDER'], file.filename)
        img.save(uploaded_img_path)
        query = gencap.get_caption(uploaded_img_path)
        answers = description_search(query)

        return render_template('search.html',
                               query_path=uploaded_img_path,
                               answers=answers)
    else:
        return render_template('search.html')


@app.route('/api/search', methods=['POST'])
def api_search():
    global gencap
    if 'query_img' not in request.files or request.files['query_img'].filename == '' or not allowed_file(
            request.files['query_img'].filename):
        return Response(response=json.dumps({'success': False, 'message': 'Uploaded image is invalid or not allowed'}),
                        status=400, mimetype="application/json")
    file = request.files['query_img']
    img = Image.open(file.stream)  # PIL image
    uploaded_img_path = os.path.join(app.config['TEMP_UPLOAD_FOLDER'], file.filename)
    img.save(uploaded_img_path)
    query = gencap.get_caption(uploaded_img_path)
    answers = description_search(query)

    return Response(response=json.dumps({'success': True, 'answers': answers}),
                    status=200, mimetype="application/json")


@app.route('/database')
def database():
    images = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '[!f_]*'))
    return render_template('database.html', database_images=images)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'photos' not in request.files:
            images = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
            return render_template('database.html', database_images=images)
        fe = FeatureExtractor()
        sift = cv2.SIFT_create()
        r = redis.Redis(host='redis')
        u = AnnoyIndex(4096, 'angular')
        u2 = AnnoyIndex(4096, 'angular')
        if os.path.exists('feature.ann'):
            u.load('feature.ann')
            all_vectors = u.get_nns_by_item(0, u.get_n_items())
            for i in all_vectors:
                u2.add_item(i, u.get_item_vector(i))

        feature_size = 128 * 128
        u3 = AnnoyIndex(feature_size, 'angular')
        u4 = AnnoyIndex(feature_size, 'angular')
        if os.path.exists('feature_sift.ann'):
            u3.load('feature_sift.ann')
            all_vectors = u3.get_nns_by_item(0, u3.get_n_items())
            for i in all_vectors:
                u4.add_item(i, u3.get_item_vector(i))

        for file in request.files.getlist('photos'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filename = uuid.uuid4().hex + '.' + filename.rsplit('.', 1)[1]

                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                img = cv2.imread(str(file_path), 0)
                img = cv2.resize(img, (224, 224))
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # blob 검출 필터 파라미터 생성 ---①
                params = cv2.SimpleBlobDetector_Params()

                # 경계값 조정 ---②
                params.minThreshold = 10
                params.maxThreshold = 240
                params.thresholdStep = 5
                # 면적 필터 켜고 최소 값 지정 ---③
                params.filterByArea = True
                params.minArea = 200

                # 컬러, 볼록 비율, 원형비율 필터 옵션 끄기 ---④
                params.filterByColor = False
                params.filterByConvexity = False
                params.filterByInertia = False
                params.filterByCircularity = False

                # 필터 파라미터로 blob 검출기 생성 ---⑤
                detector = cv2.SimpleBlobDetector_create(params)
                # 키 포인트 검출 ---⑥
                keypoints = detector.detect(img)
                # 키 포인트 그리기 ---⑦
                img_draw = cv2.drawKeypoints(img, keypoints, None, None, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # 결과 출력 ---⑧
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'f_' + filename), img_draw)

                feature = fe.extract(img=Image.open(file_path))
                feature_path = Path("./static/feature") / (filename + ".npy")  # e.g., ./static/feature/xxx.npy
                # np.save(str(feature_path), feature)
                key = hashlib.sha1(feature).hexdigest()
                if r.get(key):
                    continue
                r.set(key, filename)

                u2.add_item(u2.get_n_items(), feature.ravel())


                img = cv2.imread(str(file_path), 0)
                img = cv2.resize(img, (224, 224))
                # gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = sift.detectAndCompute(img, None)
                sift_feature_path = Path("static/sift") / (filename + ".npy")
                # np.save(str(sift_feature_path), des)

                des = des.flatten()
                if des.size < feature_size:
                    data_size = des.size
                    des = np.concatenate([des, np.zeros(feature_size - des.size, dtype='float32')])
                else:
                    data_size = feature_size

                des = des[:feature_size]
                u4.add_item(u4.get_n_items(), des)

                key = hashlib.sha1(des).hexdigest()
                r.set(key, filename)
                r.set(key + '_size', data_size)

        u2.build(10)
        u2.save('feature.ann')
        u4.build(10)
        u4.save('feature_sift.ann')

        images = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
        return render_template('database.html', database_images=images)


@app.route('/upload2', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':
        if 'photos' not in request.files:
            images = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
            return render_template('database.html', database_images=images)
        actions = []
        for file in request.files.getlist('photos'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                cap = gencap.get_caption(file_path)
                doc = {'imgurl': file_path, 'description': cap}
                actions.append(doc)
        bulk(es, actions, index="desearch", doc_type="json")
        images = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
        return render_template('database.html', database_images=images)


@app.route('/caption', methods=['GET', 'POST'])
def caption():
    if request.method == 'POST':
        if 'query_img' not in request.files or request.files['query_img'].filename == '' or not allowed_file(
                request.files['query_img'].filename):
            return render_template('caption.html')
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = os.path.join(app.config['TEMP_UPLOAD_FOLDER'], file.filename)
        img.save(uploaded_img_path)
        cap = gencap.get_caption(uploaded_img_path)
        return render_template('caption.html', caption=cap, query_path=uploaded_img_path)
    else:
        return render_template('caption.html')


@app.route('/api/caption', methods=['POST'])
def caption_api():
    if 'query_img' not in request.files or request.files['query_img'].filename == '' or not allowed_file(
            request.files['query_img'].filename):
        return Response(response=json.dumps({'success': False, 'message': 'Uploaded image is invalid or not allowed'}),
                        status=400, mimetype="application/json")
    file = request.files['query_img']
    img = Image.open(file.stream)  # PIL image
    uploaded_img_path = os.path.join(app.config['TEMP_UPLOAD_FOLDER'], file.filename)
    img.save(uploaded_img_path)
    cap = gencap.get_caption(uploaded_img_path)
    return Response(response=json.dumps({'success': True, 'caption': cap}),
                    status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
