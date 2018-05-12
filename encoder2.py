# Image set encoder.
#
# Copyright (c) 2016-  Dong Liu (dongeliu@ustc.edu.cn)
#
# For research purpose only, cannot be used in commercial products without
# permission from the author(s).

import math
import os
import zipfile

import numpy
import skimage.io
import skimage.transform

import codec

import heapq

def extend(img, grid = 8, x_ext = True, y_ext = True):
  grid = float(grid)
  width = img.shape[1]
  height = img.shape[0]
  if x_ext:
    ext_width = math.ceil(width / grid) * grid
  else:
    ext_width = width
  if y_ext:
    ext_height = math.ceil(height / grid) * grid
  else:
    ext_height = height
  ext_img = img
  if ext_width > width:
    lastCol = numpy.expand_dims(ext_img[:, -1], 1)
    tile_reps = [1, ext_width - width] + [1 for i in range(img.ndim - 2)]
    ext_img = numpy.hstack((ext_img, numpy.tile(lastCol, tile_reps)))
  if ext_height > height:
    lastRow = numpy.expand_dims(ext_img[-1, :], 0)
    tile_reps = [ext_height - height] + [1 for i in range(img.ndim - 1)]
    ext_img = numpy.vstack((ext_img, numpy.tile(lastRow, tile_reps)))
  return [ext_img, ext_width, ext_height]

# Convert image from RGB to YUV420
# According to the equations in JPEG encoder
# Y  =  0.299 * R + 0.587 * G + 0.114 * B
# Cb = -0.168735892 * R - 0.331264108 * G + 0.5 * B + CENTERJSAMPLE
# Cr =  0.5 * R - 0.418687589 * G - 0.081312411 * B + CENTERJSAMPLE
def rgb2yuv(img):
  R = img[:,:,0] / 255.0
  G = img[:,:,1] / 255.0
  B = img[:,:,2] / 255.0
  Y  =  0.299 * R + 0.587 * G + 0.114 * B
  Cb = -0.168735892 * R - 0.331264108 * G + 0.5 * B + 0.5
  Cr =  0.5 * R - 0.418687589 * G - 0.081312411 * B + 0.5
  Y = numpy.uint8(numpy.round_(Y * 255.0))
  Cb = skimage.transform.downscale_local_mean(Cb, (2,2))
  U = numpy.uint8(numpy.round_(Cb * 255.0))
  Cr = skimage.transform.downscale_local_mean(Cr, (2,2))
  V = numpy.uint8(numpy.round_(Cr * 255.0))
  return (Y, U, V)

def img2yuv(img_info, yuv_filename, force_yuv420 = False, num_blank = 0):
  [ext_img, yuv_width, yuv_height] = extend(skimage.io.imread(img_info['name']), 8)
  blank_y = 128*numpy.ones(yuv_width * yuv_height, dtype='uint8')
  blank_uv = 128*numpy.ones(yuv_width * yuv_height / 2, dtype='uint8')
  with open(yuv_filename, 'wb') as yuv_file:
    for n in range(num_blank):
      blank_y.tofile(yuv_file)
      if force_yuv420 or img_info['is_color']:
        blank_uv.tofile(yuv_file)
    if img_info['is_color']:
      yuv = rgb2yuv(ext_img)
      for i in range(3):
        yuv[i].tofile(yuv_file)
    else:
      ext_img.tofile(yuv_file)
      if force_yuv420:
        blank_uv.tofile(yuv_file)
  if force_yuv420 or img_info['is_color']:
    yuv_format = 420
  else:
    yuv_format = 400
  return (yuv_width, yuv_height, yuv_format)

#select the best 4(max_num_models) references for currrent image
def ref_select(img_info, ref_yuvs, cur_img, int_folder, max_num_models):
  new_refs = []
  ref_index = []
  # Convert this image into YUV with 0 blank images
  (yuv_width, yuv_height, yuv_format) = img2yuv(img_info, '%s%dori.yuv' % (int_folder, cur_img), True, 0)
  diff_file = '%d_diff.info' % cur_img

  oriyuv = '%s%dori.yuv' % (int_folder, cur_img)
  blockmatch_exe = 'bin\\blockmatch.exe'
  cur_ref_num = len(ref_yuvs)
  cmd = "%s %s %s %dx%dx1 16 %s %d %d" % (blockmatch_exe, oriyuv, 'x'.join(ref_yuvs), yuv_width, yuv_height, diff_file, cur_ref_num, max_num_models)
  print cmd
  os.system(cmd)

  diff_info = codec.loadDiffInfo(diff_file)
  print('Reference: %s'%ref_yuvs)
  print('Weight : %s'%diff_info)
  slist = heapq.nlargest(max_num_models, diff_info)
  # for index, text in enumerate(diff_info):
  #   if text in slist:
  #     new_refs.append(ref_yuvs[index])
  #     ref_index.append(index)
  for i in slist:
    for index, j in enumerate(diff_info):
      if j == i:
        new_refs.append(ref_yuvs[index])
        ref_index.append(index)

  new_refs.reverse()
  ref_index.reverse()
  print('Ref index: %s'%ref_index)
  print('New refereces: %s ' % new_refs)
  return [new_refs, ref_index]



def encoder(img_folder, out_file, parameters):
  # Add path
  if 'AddPath' in parameters:
    codec.add2path(parameters['AddPath'])

  # Check images
  list_files = os.listdir(img_folder)
  list_imgs = []
  for f in list_files:
    img_fname = os.sep.join([img_folder, f])
    try:
      img = skimage.io.imread(img_fname)
    except Exception:
      print('Warning: The file %s seems not an image, and thus ignored' % img_fname)
      continue
    if img.dtype != 'uint8':
      print('Warning: The file %s is not an 8-bit image, and thus ignored' % img_fname)
      continue
    if img.ndim == 2 or img.shape[2] == 1:
      is_color = False
    elif img.shape[2] == 3:
      is_color = True
    else:
      print('Warning: The file %s is neither grayscale nor RGB, and thus ignored' % img_fname)
      continue
    height = img.shape[0]
    width = img.shape[1]
    list_imgs.append({'name': img_fname, 'width': width, 'height': height, 'is_color': is_color})
  cnt_imgs = len(list_imgs)

  # Write the metadata of images
  with open('imgs.info', 'wt') as imgs_file:
    for cur_img in range(cnt_imgs):
      img_info = list_imgs[cur_img]
      (_, img_mainname) = os.path.split(img_info['name'])
      imgs_file.write('%s %d %d %d\n' % (img_mainname, img_info['width'], img_info['height'], img_info['is_color']))

  # Need an internal folder to cache some results
  int_folder = parameters.get('InternalFolder', 'enc_internal')
  int_folder.rstrip('/\\')
  try:
    os.makedirs(int_folder)
  except Exception:
    pass
  int_folder += os.sep

  # Generate coding structure
  if cnt_imgs > 1 and (not parameters.get('AllIntra', False)):
    # SIFT extraction, can be made parallel in future
    sift_exe = parameters.get('SiftExe', 'bin\\siftfeat.exe')
    for i in range(cnt_imgs):
      cmd = '%s -f %s -o %s%d.sift' % (sift_exe, list_imgs[i]['name'], int_folder, i)
      print(cmd)
      os.system(cmd)
    # Match images
    match_exe = parameters.get('MatchExe', 'bin\\match.exe')
    max_num_models = parameters.get('MaxNumModels', 1)
    photometric_dof = parameters.get('PhotometricDOF', 0)
    cmd = '%s %s %s %d %d %s%s' % (match_exe, ' '.join(['%s' % list_imgs[i]['name'] for i in range(cnt_imgs)]), ' '.join(['%s%d.sift' % (int_folder, i) for i in range(cnt_imgs)]), max_num_models, photometric_dof, int_folder, 'match.info')
    print(cmd)
    os.system(cmd)
    # Generate coding structure
    codstr_exe = parameters.get('CodStrExe', 'bin\\codstr.exe')
    cmd = '%s %s%s %s %s' % (codstr_exe, int_folder, 'match.info', 'order.info', 'pred.info')
    print(cmd)
    os.system(cmd)
  else:
    o = list(range(cnt_imgs))
    with open('order.info', 'wt') as wfile:
      wfile.write('%s' % (' '.join([str(x) for x in o])))
    with open('pred.info', 'wt'):
      pass

# Load the coding order to use
  order = codec.loadOrderInfo('order.info')

  # Load the prediction information to use
  pred_info = codec.loadPredInfo('pred.info')
  model_size = 9 + photometric_dof
  pred = [p[0:2] for p in pred_info]

  # Load the all predeiction information to use
  all_info = codec.loadAllPredInfo('enc_internal\\match.info')


  #generate the tree(
  tree = [ ]
  for i in order:
    for p in pred:
      if p[1]== i:
        tree.append((i,p[0]))
    if i==order[0]:
      tree.append((i,-1))

  #add refernece frame from parents' parents
  sptree = []
  for i in order:
    if i == order[0]:
      sptree.append((i,[-1,-1]))
    else:
      parent = tree[order.index(i)][1]
      sptree.append((i,[parent,tree[order.index(parent)][1]]))

  #generate the extend prediction information to use
  ex_pred_info = []
  for p in sptree:
    if p[1][0] != -1:
      info = all_info[p[0]+ cnt_imgs*p[1][0]]
      if isinstance(info, tuple):
        ex_pred_info.append(info)
    if p[1][1] != -1:
      info = all_info[p[0]+ cnt_imgs*p[1][1]]
      if isinstance(info, tuple):
        ex_pred_info.append(info)




  #Compress the image one by one
  for cur_img in order:
    max_num_models = parameters.get('MaxNumModels', 1)
    img_info = list_imgs[cur_img]
    ref_yuvs = []
    for p in ex_pred_info:
      if p[1] == cur_img:
        ref_img = p[0]
        ref_filename = '%s%drecon.yuv' % (int_folder, ref_img)
        if not os.access(ref_filename, os.R_OK):
          print('Warning: The required reference %s seems not available' % ref_filename)
          continue
        num_model = p[2]
        warp_exe = parameters.get('WarpExe', 'bin\\warp.exe')
        for model in range(num_model):
          cmd = '%s %s %s%dref_from%d_model%d.yuv %s %dx%dx1 %dx%dx1' % (warp_exe, ref_filename, int_folder, cur_img, ref_img, model, 'x'.join(p[3][model]), list_imgs[ref_img]['width'], list_imgs[ref_img]['height'], img_info['width'], img_info['height'])
          print(cmd)
          os.system(cmd)
          ref_yuvs.append('%s%dref_from%d_model%d.yuv' % (int_folder, cur_img, ref_img, model))
    #select 4(max_num_models) references freom ref_yuvs
    if len(ref_yuvs) > max_num_models:
      [new_refs, ref_index] = ref_select(img_info,ref_yuvs,cur_img, int_folder, max_num_models)
      for index_p, p in enumerate(ex_pred_info):
        listp = list(p)
        if listp[1] == cur_img:
          num_model = listp[2]
          for i in range(num_model):
            if i in ref_index:
              ref_index.remove(i)
            else:
              listp[3][i] = None
              listp[2]  -= 1
          listp[3] = filter(None, listp[3])
          ref_index = [index-num_model for index in ref_index]
        ex_pred_info[index_p] = tuple(listp)
      ref_yuvs = new_refs

    # Convert this image into YUV with specific number of blank images (will be replaced by references during coding)
    (yuv_width, yuv_height, yuv_format) = img2yuv(img_info, '%s%d.yuv' % (int_folder, cur_img), True, len(ref_yuvs))
    # Prepare references
    cmd = 'del /q poc*.yuv'
    print(cmd)
    os.system(cmd)
    for i in range(len(ref_yuvs)):
      for l in range(2):
        cmd = 'copy /y %s poc%dlist%dref%d.yuv'% (ref_yuvs[i], len(ref_yuvs), l, i)
        print(cmd)
        os.system(cmd)
    # Encoding by AVS2
    encoder_exe = parameters.get('EncoderExe', 'bin\\lencod.exe')
    if len(ref_yuvs) > 0:
      qp = parameters['QPB']
    else:
      qp = parameters["QPI"]
    cmd = '%s -f encoder_ldp.cfg -p InputFile=%s%d.yuv SourceWidth=%d SourceHeight=%d FramesToBeEncoded=%d QPIFrame=%d QPPFrame=%d QPBFrame=%d ReconFile=%s%drec.yuv OutputFile=%dbits.bin > %s%dencoder.log' % \
          (encoder_exe, int_folder, cur_img, yuv_width, yuv_height, len(ref_yuvs) + 1, qp, qp, qp, int_folder, cur_img, cur_img, int_folder, cur_img)
    print(cmd)
    os.system(cmd)
    codec.extract_yuv('%s%drec.yuv' % (int_folder, cur_img), '%s%drecon.yuv' % (int_folder, cur_img), len(ref_yuvs), yuv_width, yuv_height, True)


  # Write extend prediction information to use
  with open('ex_pred.info', 'wt') as ex_pred_file:
    ex_pred_file.write('%s\n' % str(model_size))
    for each_pred in ex_pred_info:
      if each_pred[2]:
        ex_pred_file.write('%s %s %s ' % (str(each_pred[0]), str(each_pred[1]), str(each_pred[2])))
      for i in each_pred[3]:
        for j in i:
          ex_pred_file.write('%s ' % str(j))
      if each_pred[2]:
        ex_pred_file.write('\n')

  # Prepare the final output
  z = zipfile.ZipFile(out_file, 'w')
  z.write('imgs.info')
  z.write('order.info')
  z.write('ex_pred.info')
  for i in range(len(list_imgs)):
    z.write('%dbits.bin' % i)
  z.close()
  cmd = 'del /q poc*.yuv *.info *bits.bin'
  print(cmd)
  os.system(cmd)


if __name__ == '__main__':
  encoder('inputimgs', 'imgset.bin', {'AllIntra' : False, 'AddPath' : 'dlls', 'MaxNumModels' : 4, 'PhotometricDOF' : 2, 'QPI' : 25, 'QPB' : 30})
