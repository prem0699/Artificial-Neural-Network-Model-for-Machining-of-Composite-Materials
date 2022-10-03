function output_denorm = denormalize(z,Z)
output_denorm=((z-0.1)*(max(Z)-min(Z))/0.8)+min(Z);
end

