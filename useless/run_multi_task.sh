# ./mmrec_rec.sh 1e-4 1 2 cosine img_sel
# ./mmrec_rec.sh 1e-4 1 2 cosine exp
# ./mmrec_rec.sh 1e-4 img_gen
# ./mmrec_rec.sh 1e-4 img_gen
# ./mmrec_rec.sh 3e-4 rec 4 3b
# ./mmrec_rec.sh 5e-5 img_gen 2 3b
# ./mmrec_rec.sh 1e-5 img_gen 2 4b-instruct
# ./mmrec_rec.sh 1e-5 img_gen 2 4b-instruct
# ./mmrec_rec.sh 2e-5 img_gen 1 4b-instruct
# ./mmrec_eval.sh 2e-5 img_gen 1 4b-instruct

# img_gen
# ./mmrec_rec.sh 1e-4 search 4 4b-instruct
# ./mmrec_rec.sh 1e-4 img_gen 1 4b-instruct
# ./mmrec_rec.sh 5e-5 img_gen 2 4b-instruct

# all other tasks together
# no semantic
# ./mmrec_all.sh 2e-5 3 4b-instruct beauty
# ./mmrec_eval.sh 2e-4 3 4b-instruct all 21 2 2
# ./mmrec_eval.sh 2e-4 3 4b-instruct all 9 2 2
# ./mmrec_all_gamma_grad.sh 5e-5 3 4b-instruct all 2 2
# ./mmrec_all_multi_gamma_grad.sh 2e-4 3 4b-instruct all 2 2

# ./mmrec_all_prefix.sh 5e-6 3 4b-instruct all rec 2 2
# ./mmrec_all_prefix.sh 1e-5 3 4b-instruct all exp 2 2

./mmrec_all_gamma_grad.sh 1e-4 1 9b all 2 4

# ./mmrec_all_gamma_grad.sh 2e-4 3 4b-instruct all 1 2
# ./mmrec_all_gamma_grad.sh 2e-4 3 4b-instruct all 2 4
# ./mmrec_all_gamma_grad.sh 2e-4 3 4b-instruct all 2 2
# ./mmrec_all_gamma_grad.sh 2e-4 3 4b-instruct all 5 1
# ./mmrec_all.sh 2e-4 3 4b-instruct all
# ./mmrec_all_multi.sh 2e-4 3 4b-instruct all
# ./mmrec_all_semantic.sh 2e-4 3 4b-instruct all
# ./mmrec_all_semantic.sh 5e-5 3 4b-instruct all
# ./mmrec_eval.sh 5e-4 3 4b-instruct all 29
# ./mmrec_eval.sh 2e-4 3 4b-instruct all 29
# ./mmrec_eval.sh 1e-4 3 4b-instruct all
# ./mmrec_eval.sh 2e-5 2 4b-instruct
# semantic
