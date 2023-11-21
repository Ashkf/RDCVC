package com.refiner.services.airduct.model;

import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import org.springframework.data.mongodb.core.mapping.Document;

import java.util.Date;
import java.util.List;
import java.util.Map;

@Document
@ApiModel("全部计算结果")
public class AirductContext extends UserItem{

    @ApiModelProperty("状态:wait, success, fail ")
    private String status;

    @ApiModelProperty("错误,计算错误，网络错误，其他错误")
    private String error;

    @ApiModelProperty("请求送风系统ID")
    private String requestSupplySystemId;

    @ApiModelProperty("请求回风系统ID")
    private String requestReturnSystemId;

    @ApiModelProperty("请求排风系统ID")
    private String requestExhaustSystemId;

    @ApiModelProperty("送风系统")
    private RevitSystem supplySystem;

    @ApiModelProperty("回风系统")
    private RevitSystem returnSystem;

    @ApiModelProperty("排风系统")
    private RevitSystem exhaustSystem;

    @ApiModelProperty("送风系统压降")
    private Double supplyImpedance;

    @ApiModelProperty("回风系统压降")
    private Double returnImpedance;

    @ApiModelProperty("排风系统压降")
    private Double exhaustImpedance;

    @ApiModelProperty("排风系统压降")
    private Map<String,Double> idToPathRootImpedance;

    @ApiModelProperty("风阀配置快照")
    private List<DamperConfig> damperConfigsSnapShot;

    @ApiModelProperty("风机配置快照")
    private FanConfig fanConfigSnapShot;

    @ApiModelProperty("门缝系数配置快照")
    private List<RelationCoEfficientConfig> relationCoEfficientConfigSnapShots;

    @ApiModelProperty("房间系统快照")
    private List<RevitRoom> revitRoomsSnapShot;
  //  private List<RevitElement> revitElementsSnapShot;
//    private List<RevitSystem> revitSystemsSnapShot;

    @ApiModelProperty("送风末端ID到阻抗")
    private Map<String,Double> supplyTerminalIdToImpedance;

    @ApiModelProperty("回风末端ID到阻抗")
    private Map<String,Double> returnTerminalIdToImpedance;

    @ApiModelProperty("排风末端ID到阻抗")
    private Map<String,Double> exhaustTerminalIdToImpedance;

    @ApiModelProperty("送风末端ID到风量")
    private Map<String,Double> supplyTerminalIdToWindVolume;

    @ApiModelProperty("回风末端ID到风量")
    private Map<String,Double> returnTerminalIdToWindVolume;

    @ApiModelProperty("排风末端ID到风量")
    private Map<String,Double> exhaustTerminalIdToWindVolume;

    @ApiModelProperty("送风末端标记名称到阻抗")
    private Map<String,Double> supplyMarkTerminalToImpedance;

    @ApiModelProperty("回风末端标记名称到阻抗")
    private Map<String,Double> returnMarkTerminalToImpedance;

    @ApiModelProperty("排风末端标记名称到阻抗")
    private Map<String,Double> exhaustMarkTerminalToImpedance;

    @ApiModelProperty("送风末端标记名称到风量")
    private Map<String,Double> supplyMarkTerminalToWindVolume;

    @ApiModelProperty("回风末端标记名称到风量")
    private Map<String,Double> returnMarkTerminalToWindVolume;

    @ApiModelProperty("排风末端标记名称到风量")
    private Map<String,Double> exhaustMarkTerminalToWindVolume;

    @ApiModelProperty("送风末端ID到标记")
    private Map<String,String> supplyTerminalIdToSupplyMark;
    @ApiModelProperty("回风末端ID到标记")
    private Map<String,String> returnTerminalIdToReturnMark;
    @ApiModelProperty("排风末端ID到标记")
    private Map<String,String> exhaustTerminalIdToExhaustMark;

    @ApiModelProperty("计算开始时间")
    private Date startTime;

    @ApiModelProperty("计算结束时间")
    private Date endTime;

    //返回
    @ApiModelProperty("送风压降H_S")
    private Double windVolumePressureDropHs;

    @ApiModelProperty("回风压降H_R")
    private Double returnPressureDropHr;

    @ApiModelProperty("AHU内部压降H_3")
    private Double ahuPressureDropH3;

    @ApiModelProperty("送风机压力提升值H_AHU")
    private Double windVolumePressureDropAhu;

    @ApiModelProperty("新风机压力提升值H_MAU")
    private Double newPressureDropMau;

    @ApiModelProperty("新风机入口端压力值H_in")
    private Double newPressureDropHin;

    @ApiModelProperty("总送风量Q_S")
    private Double totalWindVolumeQs;

    @ApiModelProperty("新风量Q_F")
    private Double newWindVolumeQf;

    @ApiModelProperty("总回风量Q_R")
    private Double totalNewWindVolumeQr;

    @ApiModelProperty("排风机压头")
    private Double pressureHe;

    @ApiModelProperty("排风量")
    private Double exhaustQe;

    @ApiModelProperty("渗凤量")
    private Map<String,Double> roomNameToRemainWindVolume;
    @ApiModelProperty("房间到排风量")
    private Map<String,Double> roomNameToSupplyWindVolume;
    @ApiModelProperty("房间到回风量")
    private Map<String,Double> roomNameToReturnWindVolume;
    @ApiModelProperty("房间到排风量")
    private Map<String,Double> roomNameToExhaustWindVolume;
    @ApiModelProperty("房间到设计压力")
    private Map<String,Integer> roomNameToDesignPressure;
    @ApiModelProperty("房间到实际压力")
    private Map<String,Double> roomNameToRealPressure;
    @ApiModelProperty("房间到送风风阀配置")
    private Map<String,List<DamperConfig>> roomNameToSupplyDamperConfigs;
    @ApiModelProperty("房间到回风风阀配置")
    private Map<String,List<DamperConfig>> roomNameToReturnDamperConfigs;

    @ApiModelProperty("房间到排风风阀配置")
    private Map<String,List<DamperConfig>> roomNameToExhaustDamperConfigs;

   
}
