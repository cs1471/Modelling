function printCellString(cellOfString)

[numRows,numCols] = size(cellOfString);
for r=1:numRows
    for c=1:numCols
        disp(cellOfString{r,c})
    end
end


end