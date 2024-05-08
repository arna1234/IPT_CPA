# DIR= where is IPT_CPA.py?

DIR=../..   # where is IPT_CPA.py?
prog=IPT_disord_box_bethe_sigmaloop.py
T=0.02
U=2.0
delteeno=0.0007
dis_in=0.8
dis_fin=2.0
Ndis=4
#
# U loop
#
for dis in `awk 'BEGIN { for( il=0; il<='${Ndis}-1'; il++ ) printf ("%.1f\n",'${dis_in}'+il*('${dis_fin}'-('${dis_in}'))/('${Ndis}'-1)) }'` ; do
 echo " "
 echo " ...running: W= " ${dis}
 echo " "
 data_dir="T_${T}_U_${U}_delta_${delteeno}_box_${dis}" # a leading string for directories
 mkdir -p ${data_dir}
 cp ipt.parameters ${data_dir}/
# cp sigma_00.dat ${data_dir}/
 cd ${data_dir}
 python ${DIR}/${prog} ${U} 0 ${T} ${delteeno} ipt.parameters ${dis} > ${data_dir}.rpt 
 cd ..
done
