// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimoraLog {
    struct Prediction {
        uint256 timestamp;
        string mineId;
        uint8 riskLevel;
        string dataHash;
        address validator;
    }

    Prediction[] public predictions;
    mapping(address => bool) public validators;

    event PredictionLogged(uint256 indexed id, string mineId, uint8 riskLevel, address validator);

    constructor() {
        validators[msg.sender] = true; // deployer (you) is validator
    }

    function addValidator(address _validator) external {
        require(validators[msg.sender], "Not validator");
        validators[_validator] = true;
    }

    function logPrediction(string memory mineId, uint8 riskLevel, string memory dataHash) external {
        require(validators[msg.sender], "Not validator");
        predictions.push(Prediction(block.timestamp, mineId, riskLevel, dataHash, msg.sender));
        emit PredictionLogged(predictions.length - 1, mineId, riskLevel, msg.sender);
    }

    function getPredictionCount() external view returns (uint256) {
        return predictions.length;
    }
}