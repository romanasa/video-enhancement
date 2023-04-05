python PrepareYUVsequences.py --raw_path ../org/foreman.yuv --com_path ../dec/foreman_dec_h265.yuv --frame_num 300 --H 288 --W 352 
python MWGANYUV.py -opt test_MWGAN_PSNR.yml 
python StoreFinalYUVsequence.py --raw_path ../org/foreman.yuv --com_path ../dec/foreman_dec_h265.yuv --enh_path ../enh/H265/foreman_enh_mwgan.yuv --frame_num 300 --H 288 --W 352 
pause