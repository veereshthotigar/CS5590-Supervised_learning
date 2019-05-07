% Mode B
PCA = importdata('PCAnormalizedscore.mat');
LDA = importdata('LDAnormalizedscore.mat');

%Create the label matrix
zeroMatrix = zeros(5);
oneMatrix = ones(5);
labelsMain = [];
labelCount = 1;
generalMat = [];
for i=1:40 
    generalMat = [];
    for j=1:40
        if(labelCount==j)
            generalMat = horzcat(generalMat,zeroMatrix);
        else
            generalMat = horzcat(generalMat,oneMatrix);
        end
    end
    labelsMain = vertcat(labelsMain,generalMat);
    labelCount = labelCount + 1;
end


[roc_PCA, EER_PCA, area_PCA, EERthr_PCA, ALLthr_PCA, d_PCA, gen_PCA, imp_PCA] = ezroc3(PCA,labelsMain,2,'',1);
GAR_PCA = roc_PCA(1,:);
FAR_PCA = roc_PCA(2,:);

PCA_EERthr_ind = find(ALLthr_PCA==EERthr_PCA);
PCA_GAR_at_thr = GAR_PCA(PCA_EERthr_ind);
PCA_FAR_at_thr = FAR_PCA(PCA_EERthr_ind);
PCA_accuracy = 1-EER_PCA;

[roc_LDA, EER_LDA, area_LDA, EERthr_LDA, ALLthr_LDA, d_LDA, gen_LDA, imp_LDA] = ezroc3(LDA,labelsMain,2,'',1);
GAR_LDA = roc_PCA(1,:);
FAR_LDA = roc_PCA(2,:);

LDA_EERthr_ind = find(ALLthr_LDA==EERthr_LDA);
LDA_GAR_at_thr = GAR_LDA(LDA_EERthr_ind);
LDA_FAR_at_thr = FAR_LDA(LDA_EERthr_ind);
LDA_accuracy = 1-EER_LDA;

PCA_threshold = EERthr_PCA;
LDA_threshold = EERthr_LDA;

PCA_Decision = [];
for c = 1:200
    for r = 1:200 
	PCA_Decision(c,r) = (PCA(c,r) > PCA_threshold);
    end
end

LDA_Decision = [];
for c = 1:200
    for r = 1:200 
	LDA_Decision(c,r) = (LDA(c,r) > LDA_threshold);
    end
end
rng(0)
decision_fusion = [];
for c = 1:200
    for r = 1:200 
        if(LDA_Decision(c,r) == PCA_Decision(c,r))
            decision_fusion(c,r) = LDA_Decision(c,r);
        else
            decision_fusion(c,r) = round(rand); % In the case of a tie, decide randomly
        end
    end
end
total_FA = 0;
total_GA = 0;
total_correct = 0;
total = 40000;
total_GR = 0;
total_FR = 0;

for c = 1:200
    for r = 1:200 
        total_correct = total_correct+(decision_fusion(c,r) == LDA_Decision(c,r));
        if(LDA_Decision(c,r) == 0)
            if(decision_fusion(c,r) == 0)
                total_GA = total_GA + 1;
            else
                total_FA = total_FA + 1;
            end
        else
            if(decision_fusion(c,r) == 1)
                total_GR = total_GR + 1;
            else
                total_FR = total_FR + 1;
            end
        end
    end
end

FRR = total_FR/(total_FR+total_GR);
decision_fusion_GAR = 1-FRR;
decision_fusion_FAR = total_FA/(total_FA+total_GA);
decision_fusion_Accuracy = total_correct/total;